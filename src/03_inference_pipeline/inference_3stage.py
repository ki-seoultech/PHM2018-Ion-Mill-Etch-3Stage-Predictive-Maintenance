"""
3-Stage real-time inference pipeline

  Stage 0  Z-Score safety gate: max |z| of 4 core sensors < 1.5 -> SAFE, no model call
  Stage 1  XGBoost fault-cause classifier -> F1 / F2 / F3
  Stage 2  fault-specific RUL regressor (HGBM + GRU/LSTM dynamic ensemble)

The simulation block at the bottom feeds mock sensor scenarios (Z-scaled) to check
the end-to-end flow. Models themselves were trained/validated on PHM 2018 data.

Input : saved weights from 02_fault_classifier and 03_inference_pipeline
Output: SAFE or (fault index, RUL seconds) + console alarm
"""

# =========================================================
# 🚀 3-Stage MLOps 실시간 추론(Inference) 파이프라인
#    + Z-Score 이상 탐지 게이트 (안전 구간 자동 차단)
#
# ▶ 동작 흐름
#   실시간 센서 데이터 (Z-Score 스케일링 완료됨) 입력
#       ↓
#   [Stage 0] L/H 안전 게이트: 핵심 센서 편차가 임계값(1.5) 미만인가?
#       ├── YES (안전) → 파이프라인 즉시 종료 (전력 낭비 및 억지 예측 방지)
#       └── NO  (위험) → 다중 분류기 가동
#             ↓
#   [Stage 1] 다중분류기 → F1/F2/F3 고장 원인 판별
#             ↓
#   [Stage 2] 회귀 모델 라우팅 → 맞춤형 RUL 예측 → 경보 출력
# =========================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import logging

# TensorFlow 경고(Retracing 등) 메시지 숨김
tf.get_logger().setLevel(logging.ERROR)

# TF와 XGBoost가 GPU 메모리를 나눠 쓰도록 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

# =========================================
# 0. Z-Score 안전 게이트 (L/H 필터) 설정
# =========================================
# 핵심 센서 4종의 Z-Score 절대값이 모두 이 수치 미만이면 '안전(L)'으로 간주합니다.
# (통상적으로 Z-Score 1.5 ~ 2.0 사이를 정상 범주의 한계로 봅니다)
Z_SCORE_THRESHOLD = 1.5

# =========================================
# 1. 저장된 모든 모델 & 전처리기 일괄 로드
# =========================================
print("[INFO] Model Registry에서 AI 가중치를 로드하고 있습니다...")
MODEL_DIR = '/content/drive/MyDrive/PdM_Models_Regression/'

# Stage 1: 다중분류기 로드
clf_prep  = joblib.load(f'{MODEL_DIR}classifier_preprocessor.pkl')
clf_model = joblib.load(f'{MODEL_DIR}champion_classifier.pkl')

# Stage 2: 고장 모드별 회귀 챔피언 로드 (TF 예측은 CPU에서 안전하게 처리)
with tf.device('/CPU:0'):
    dl_f1 = tf.keras.models.load_model(f'{MODEL_DIR}F1_GRU_model.h5',  compile=False)
    dl_f2 = tf.keras.models.load_model(f'{MODEL_DIR}F2_LSTM_model.h5', compile=False)
    dl_f3 = tf.keras.models.load_model(f'{MODEL_DIR}F3_GRU_model.h5',  compile=False)

reg_models = {
    0: {"hgbm": joblib.load(f'{MODEL_DIR}F1_hgbm_model.pkl'),
        "dl_prep": joblib.load(f'{MODEL_DIR}F1_preprocessor_dl.pkl'),
        "dl": dl_f1, "max_ttf": 14000},
    1: {"hgbm": joblib.load(f'{MODEL_DIR}F2_hgbm_model.pkl'),
        "dl_prep": joblib.load(f'{MODEL_DIR}F2_preprocessor_dl.pkl'),
        "dl": dl_f2, "max_ttf": 14000},
    2: {"hgbm": joblib.load(f'{MODEL_DIR}F3_hgbm_model.pkl'),
        "dl_prep": joblib.load(f'{MODEL_DIR}F3_preprocessor_dl.pkl'),
        "dl": dl_f3, "max_ttf": 15000},
}
FAULT_NAMES = {
    0: "F1 (Flowcool 압력 과다)",
    1: "F2 (Flowcool 누수)",
    2: "F3 (압력 저하 Limit)"
}
print("✅ 모든 챔피언 모델 로드 완료! (추론 준비 끝)")
print(f"   안전 게이트(L/H) 임계값: Z-Score < {Z_SCORE_THRESHOLD}")

# =========================================
# 2. 실시간 피처 추출 함수
# =========================================
def extract_realtime_features(df, sensor_cols, w=10):
    df = df.copy()
    for col in sensor_cols:
        roll = df[col].rolling(window=w, min_periods=2)
        df[f"{col}_mean"]  = roll.mean()
        df[f"{col}_std"]   = roll.std()
        df[f"{col}_max"]   = roll.max()
        df[f"{col}_rms"]   = roll.apply(lambda x: np.sqrt(np.mean(np.square(x))), raw=True)
        df[f"{col}_skew"]  = roll.skew()
        df[f"{col}_kurt"]  = roll.kurt()
        df[f"{col}_peak_factor"]     = df[f"{col}_max"] / (df[f"{col}_rms"] + 1e-8)
        mean_abs = roll.apply(lambda x: np.mean(np.abs(x)), raw=True)
        df[f"{col}_waveform_factor"] = df[f"{col}_rms"] / (mean_abs + 1e-8)
        df[f"{col}_pulse_factor"]    = df[f"{col}_max"] / (mean_abs + 1e-8)
        mean_sqrt_abs = roll.apply(lambda x: np.mean(np.sqrt(np.abs(x))), raw=True)
        df[f"{col}_margin_factor"]   = df[f"{col}_max"] / (np.square(mean_sqrt_abs) + 1e-8)
        df[f"{col}_diff"]            = df[col].diff()
        df[f"{col}_pct_change"]      = df[col].pct_change()
        df[f"{col}_mean_diff"]       = df[col] - df[f"{col}_mean"]
        df[f"{col}_ema"]             = df[col].ewm(span=5, adjust=False).mean()
        df[f"{col}_diff_smooth"]     = df[f"{col}_diff"].rolling(5, min_periods=1).mean()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.bfill(inplace=True)
    return df

# =========================================
# 3. 핵심 추론 라우터 — L/H 안전 게이트 포함
# =========================================
def predict_rul(streaming_df):
    """
    최근 15스텝(140초 단위) 센서 데이터 버퍼를 받아
    L/H 판별 → 고장 원인 판별 → RUL 예측까지 수행합니다.
    """
    SEQ_LEN = 15
    if len(streaming_df) < SEQ_LEN:
        print("⚠️  데이터 버퍼 부족 (최소 15스텝 필요)")
        return "SAFE"

    # ── 피처 추출 ──
    core_sensors = ["FLOWCOOLPRESSURE", "IONGAUGEPRESSURE",
                    "ETCHBEAMCURRENT",  "FLOWCOOLFLOWRATE"]
    live_df      = extract_realtime_features(streaming_df, core_sensors)
    current_state = live_df.iloc[[-1]].copy()

    # ── [Stage 0] L/H 안전 게이트 (Z-Score 기반) ──
    # 핵심 센서들의 현재 Z-Score 값을 가져와 가장 크게 튀는 편차를 확인합니다.
    core_z_scores = current_state[core_sensors].values[0]
    max_deviation = np.max(np.abs(core_z_scores))
    
    print("\n" + "─" * 52)
    print(f"  [Stage 0] 안전 게이트 │ 최대 센서 편차 = {max_deviation:.2f} (임계값: {Z_SCORE_THRESHOLD})")
    
    if max_deviation < Z_SCORE_THRESHOLD:
        print(f"  게이트 판정 │ ✅ SAFE (장비 정상 가동 중. 연산 종료)")
        print("─" * 52)
        return "SAFE"
        
    print(f"  게이트 판정 │ 🚨 DANGER (이상 징후 포착! 다중 분류기 가동)")
    print("─" * 52)

    # ── [Stage 1] 다중분류기 추론 (게이트 통과 시에만 실행) ──
    current_prep  = clf_prep.transform(current_state)
    fault_proba   = clf_model.predict_proba(current_prep)[0]   # shape: (3,)
    fault_idx     = int(np.argmax(fault_proba))
    top_prob      = float(fault_proba[fault_idx])

    print(f"  분류기 출력 │ F1: {fault_proba[0]:.3f} │ F2: {fault_proba[1]:.3f} │ F3: {fault_proba[2]:.3f}")
    print(f"  원인 판별   │ {FAULT_NAMES[fault_idx]} (확률 {top_prob*100:.1f}%)")
    print("─" * 52)

    # ── [Stage 2] 회귀 모델 라우팅 → RUL 예측 ──
    champion = reg_models[fault_idx]
    max_ttf  = champion["max_ttf"]

    # HGBM 추론
    g_pred = np.expm1(champion["hgbm"].predict(current_state)[0]).clip(0, max_ttf)

    # DL 추론 (15스텝 시퀀스)
    dl_input   = champion["dl_prep"].transform(live_df.iloc[-SEQ_LEN:])
    seq_input  = np.expand_dims(dl_input, axis=0)   # (1, 15, n_features)
    with tf.device('/CPU:0'):
        dl_pred = np.expm1(
            champion["dl"].predict(seq_input, verbose=0)[0][0]
        ).clip(0, max_ttf)

    # 동적 앙상블
    alpha     = np.clip((max_ttf - dl_pred) / (max_ttf * 0.8), 0.1, 0.9)
    final_rul = alpha * dl_pred + (1 - alpha) * g_pred

    # 시/분/초 변환 출력
    hours   = int(final_rul // 3600)
    minutes = int((final_rul % 3600) // 60)
    seconds = int(final_rul % 60)

    print(f"\n🚨 [최종 알람] {FAULT_NAMES[fault_idx]} 발생 위험!")
    print(f"⏳ [RUL 예측] 약 {hours}시간 {minutes}분 {seconds}초 뒤 고장 예상. 즉각 점검 요망.")

    return fault_idx, final_rul


# =========================================
# 4. 실전 시뮬레이션 테스트 (모든 고장 모드 검증)
# =========================================

# 공통 정상 베이스라인 데이터 (Z-Score 스케일)
mock_safe = pd.DataFrame({
    "FLOWCOOLPRESSURE":        np.random.normal(0, 0.2, 20), 
    "IONGAUGEPRESSURE":        np.random.normal(0, 0.2, 20),
    "ETCHBEAMCURRENT":         np.random.normal(0, 0.2, 20),
    "FLOWCOOLFLOWRATE":        np.random.normal(0, 0.2, 20),
    "ACTUALROTATIONANGLE":     np.random.normal(0, 0.2, 20),
    "ETCHSOURCEUSAGE":         np.random.normal(0, 0.2, 20),
    "ACTUALSTEPDURATION":      np.random.normal(0, 0.2, 20),
    "recipe_step":             np.zeros(20),
    "ETCHAUXSOURCETIMER":      np.zeros(20),
    "ETCHBEAMVOLTAGE":         np.random.normal(0, 0.2, 20),
    "runnum":                  np.random.normal(0, 0.2, 20),
    "FIXTURETILTANGLE":        np.zeros(20),
    "ETCHSUPPRESSORVOLTAGE":   np.random.normal(0, 0.2, 20),
    "ETCHAUX2SOURCETIMER":     np.zeros(20),
    "ETCHPBNGASREADBACK":      np.random.normal(0, 0.2, 20),
    "ROTATIONSPEED":           np.random.normal(0, 0.2, 20),
    "ETCHGASCHANNEL1READBACK": np.random.normal(0, 0.2, 20),
    "ETCHSUPPRESSORCURRENT":   np.random.normal(0, 0.2, 20),
    "recipe":                  ["Recipe_A"] * 20,
    "stage":                   ["Stage_1"]  * 20,
})

print("\n" + "=" * 52)
print(" 시뮬레이션 1 — 완전 안전 구간 (정상 데이터)")
print("=" * 52)
result_safe = predict_rul(mock_safe)

print("\n" + "=" * 52)
print(" 시뮬레이션 2 — 위험 구간 (F1: 압력 과다 징후 모사)")
print("=" * 52)
mock_f1 = mock_safe.copy()
# F1 (Pressure Too High) -> 압력 Z-score 양수 극단적 폭증, 유량 약간 증가
mock_f1["FLOWCOOLPRESSURE"] = np.linspace(0, 8, 20) + np.random.normal(0, 0.1, 20)
mock_f1["FLOWCOOLFLOWRATE"] = np.linspace(0, 2, 20) + np.random.normal(0, 0.1, 20)
result_f1 = predict_rul(mock_f1)

print("\n" + "=" * 52)
print(" 시뮬레이션 3 — 위험 구간 (F2: 냉각수 누수 징후 모사)")
print("=" * 52)
mock_f2 = mock_safe.copy()
# F2 (Leak) -> 압력은 덜 떨어지는데, 냉각수 유량(Flowrate)만 극단적으로 급감하는 패턴 모사
mock_f2["FLOWCOOLFLOWRATE"] = np.linspace(0, -8, 20) + np.random.normal(0, 0.1, 20)
mock_f2["FLOWCOOLPRESSURE"] = np.linspace(0, -1, 20) + np.random.normal(0, 0.1, 20)
result_f2 = predict_rul(mock_f2)

print("\n" + "=" * 52)
print(" 시뮬레이션 4 — 위험 구간 (F3: 압력 저하 징후 모사)")
print("=" * 52)
mock_f3 = mock_safe.copy()
# F3 (Pressure Dropped Below Limit) -> 유량은 유지되거나 덜 떨어지나, 압력 Z-score만 극단적 음수 폭락
mock_f3["FLOWCOOLPRESSURE"] = np.linspace(0, -8, 20) + np.random.normal(0, 0.1, 20)
mock_f3["FLOWCOOLFLOWRATE"] = np.linspace(0, -1, 20) + np.random.normal(0, 0.1, 20)
result_f3 = predict_rul(mock_f3)
