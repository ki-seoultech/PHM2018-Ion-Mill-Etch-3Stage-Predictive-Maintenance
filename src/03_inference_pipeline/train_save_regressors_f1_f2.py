"""
Train and export champion RUL regressors for F1 / F2

- F1: 140 s / 14,000 s / GRU ensemble, F2: 140 s / 14,000 s / LSTM ensemble
- Saves HGBM pipeline, DL preprocessor, Keras model, Isotonic dict per fault mode

Input : ALL_M01_MASTER_labeled.csv
Output: F1_*, F2_* model files in SAVE_DIR
"""

# =========================================================
# V8 Multi-Fault Auto-Tuning (F1 & F2 최적 챔피언 고정 & 가중치 저장)
# =========================================================
import os
# CUDA_ERROR_INVALID_HANDLE 방지: 회귀 학습은 CPU로 수행
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
import time
import gc
import itertools
import joblib  # 모델 저장

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression

# Colab Drive Mount
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

# =========================================
# 1. 환경 설정 및 고장 모드(Target) 정의
# =========================================
CSV_PATH = '/content/drive/MyDrive/ALL_M01_MASTER_labeled.csv'
SAVE_DIR = '/content/drive/MyDrive/PdM_Models_Regression/' # 💾 가중치 저장 폴더
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 15

# 고장모드별 타깃 컬럼
TARGET_DICT = {
    "F1": "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "F2": "TTF_Flowcool leak"
}

# V8 탐색으로 찾은 최적 파라미터 고정
FIXED_PARAMS = {
    "F1": {"rate": 140, "ttf": 14000, "mod": "GRU"},
    "F2": {"rate": 140, "ttf": 14000, "mod": "LSTM"}
}

def phm_score(y_true, y_pred):
    return np.sum(np.exp(-0.001 * y_true) * np.abs(y_true - y_pred))

# =========================================
# 2. 대용량 데이터 로드 (정답지 보호 & 기본 필터)
# =========================================
print(f"\n[INFO] 대용량 마스터 데이터 로드 중... (F1, F2 동시 로드)")
def load_filter(col):
    # 1. 정답이 직접 노출되는 label·risk 컬럼 제거
    if 'label' in col.lower() or 'risk' in col.lower():
        return False
        
    # 2. 메타데이터 및 분석에 무의미한 타이머 제거
    v6_drop_cols = ["ROTATIONSPEED", "Tool", "Lot", "runnum",
                    "ETCHAUX2SOURCETIMER", "ETCHSOURCEUSAGE",
                    "ETCHAUXSOURCETIMER", "ACTUALSTEPDURATION"]
    if col in v6_drop_cols:
        return False

    # 3. 그 외(TTF 정답 컬럼, 센서 데이터)는 모두 로드
    return True

df_raw = pd.read_csv(CSV_PATH, usecols=load_filter)

# 유효 구간 필터링 (FIXTURESHUTTERPOSITION == 1)
if "FIXTURESHUTTERPOSITION" in df_raw.columns:
    df_raw = df_raw[df_raw['FIXTURESHUTTERPOSITION'] == 1].copy()
    df_raw.drop(columns=['FIXTURESHUTTERPOSITION'], inplace=True)

print(f"       ✅ 글로벌 데이터 로드 완료! (총 {len(df_raw):,} 행)")

# =========================================
# 3. 물리적 Feature Extraction 함수
# =========================================
def add_reference_features(df, sensor_cols, w=10):
    df = df.copy()
    g = df.groupby("episode_id", sort=False)
    for col in sensor_cols:
        roll = g[col].rolling(window=w, min_periods=2)
        df[f"{col}_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{col}_std"] = roll.std().reset_index(level=0, drop=True)
        df[f"{col}_max"] = roll.max().reset_index(level=0, drop=True)
        df[f"{col}_rms"] = roll.apply(lambda x: np.sqrt(np.mean(np.square(x))), raw=True).reset_index(level=0, drop=True)
        df[f"{col}_skew"] = roll.skew().reset_index(level=0, drop=True)
        df[f"{col}_kurt"] = roll.kurt().reset_index(level=0, drop=True)
        df[f"{col}_peak_factor"] = df[f"{col}_max"] / (df[f"{col}_rms"] + 1e-8)

        mean_abs = roll.apply(lambda x: np.mean(np.abs(x)), raw=True).reset_index(level=0, drop=True)
        df[f"{col}_waveform_factor"] = df[f"{col}_rms"] / (mean_abs + 1e-8)
        df[f"{col}_pulse_factor"] = df[f"{col}_max"] / (mean_abs + 1e-8)
        mean_sqrt_abs = roll.apply(lambda x: np.mean(np.sqrt(np.abs(x))), raw=True).reset_index(level=0, drop=True)
        df[f"{col}_margin_factor"] = df[f"{col}_max"] / (np.square(mean_sqrt_abs) + 1e-8)
        
        # [추가 피처 1] 변화량 및 변화율
        df[f"{col}_diff"] = g[col].diff()
        df[f"{col}_pct_change"] = g[col].pct_change()
        
        # [추가 피처 2] 국소적 이상치 감지 (Mean 편차)
        df[f"{col}_mean_diff"] = df[col] - df[f"{col}_mean"]
        
        # [추가 피처 3] 지수이동평균(EMA) 스무딩
        df[f"{col}_ema"] = g[col].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    
    g_new = df.groupby("episode_id", sort=False)
    for col in sensor_cols:
        df[f"{col}_diff_smooth"] = g_new[f"{col}_diff"].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

    # 안전장치: pct_change 연산 시 발생할 수 있는 무한대(inf) 값을 NaN으로 변환 후 처리
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

core_sensors = ["FLOWCOOLPRESSURE", "IONGAUGEPRESSURE", "ETCHBEAMCURRENT", "FLOWCOOLFLOWRATE"]
core_sensors = [c for c in core_sensors if c in df_raw.columns]

# =========================================
# 4. 핵심 파이프라인 (타깃 변경 지원 및 정보 누수 차단)
# =========================================
def run_pipeline(fault_name, target_col, downsample_sec, max_ttf, model_type):
    print(f"\n" + "="*65)
    print(f" 🚀 [{fault_name}] 압축: {downsample_sec}초 | 상한선: {max_ttf}초 | 모델: {model_type}")
    print("="*65)
    start_time_all = time.time()

    # -------------------------------------
    # A. 타겟 전용 데이터셋 분리 및 에피소드 생성
    # -------------------------------------
    # 해당 타겟(현재 진행중인 고장 모드)의 결측치가 없는 깔끔한 데이터만 남김
    df = df_raw.dropna(subset=[target_col]).copy()
    df.sort_values(['Unit', 'time'], inplace=True)
    
    df['ttf_diff'] = df[target_col].diff()
    df['is_new_ep'] = (df['ttf_diff'] > 0) | (df['Unit'] != df['Unit'].shift())
    df['episode_id'] = df['is_new_ep'].cumsum().astype(int)
    df.drop(columns=['ttf_diff', 'is_new_ep'], inplace=True)

    df['time_bin'] = df['time'] // downsample_sec

    cat_cols = ['recipe', 'stage']
    ignore_cols = cat_cols + ['Unit', 'time', 'time_bin', 'episode_id', target_col, 'Tool', 'Lot', 'runnum', 'ROTATIONSPEED']
    num_cols = [c for c in df.columns if c not in ignore_cols and df[c].dtype != 'object']

    agg_dict = {'time': 'last', target_col: 'min'}
    for c in cat_cols:
        if c in df.columns: agg_dict[c] = 'last'
    for c in num_cols:
        agg_dict[c] = 'mean'

    df_down = df.groupby(['Unit', 'episode_id', 'time_bin']).agg(agg_dict).reset_index()
    print(f"  [1] 타겟 분리 및 {downsample_sec}초 다운샘플링 완료: {len(df_down):,} 행 압축")

    # -------------------------------------
    # B. 물리적 Feature Extraction
    # -------------------------------------
    df_down = add_reference_features(df_down, core_sensors, w=10)
    df_down.bfill(inplace=True)

    # -------------------------------------
    # C. Train/Valid Split
    # -------------------------------------
    episodes_per_unit = df_down.groupby('Unit')['episode_id'].unique()
    valid_eps = [eps[-1] for eps in episodes_per_unit if len(eps) > 1]

    train_part = df_down[~df_down['episode_id'].isin(valid_eps)].copy()
    valid_part = df_down[df_down['episode_id'].isin(valid_eps)].copy()

    # -------------------------------------
    # D. Unit-Aware Scaling & 정보 누수(Data Leakage) 차단
    # -------------------------------------
    # 다른 고장모드의 수명 컬럼(TTF_*)은 입력 피처에서 제외
    drop_cols = ["time", "time_bin", "episode_id", "ROTATIONSPEED", "Tool", "Lot", "runnum"]
    feature_cols = [c for c in train_part.columns if c not in drop_cols and c != 'Unit' and not c.startswith('TTF_')]
    
    categorical_cols = [c for c in feature_cols if train_part[c].dtype == "object" or c in cat_cols]
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]

    for unit in df_down['Unit'].unique():
        scaler = StandardScaler()
        u_train = train_part['Unit'] == unit
        u_valid = valid_part['Unit'] == unit

        if u_train.sum() > 0:
            train_part.loc[u_train, numerical_cols] = scaler.fit_transform(train_part.loc[u_train, numerical_cols])
            if u_valid.sum() > 0:
                valid_part.loc[u_valid, numerical_cols] = scaler.transform(valid_part.loc[u_valid, numerical_cols])

    X_train, y_train_raw = train_part[feature_cols], train_part[target_col].values
    X_valid, y_valid_raw = valid_part[feature_cols], valid_part[target_col].values

    y_train_capped = np.clip(y_train_raw, 0, max_ttf)
    y_valid_capped = np.clip(y_valid_raw, 0, max_ttf)
    y_train_reg = np.log1p(y_train_capped)

    # -------------------------------------
    # E. HGBM 베이스라인 학습 & 💾 모델 저장
    # -------------------------------------
    print("  [2] HGBM 베이스라인 학습 및 저장 중...")
    preprocessor_ml = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical_cols)
    ])

    gbm_model = Pipeline([
        ("prep", preprocessor_ml),
        ("model", HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_iter=250, random_state=42))
    ])

    gbm_model.fit(X_train, y_train_reg)
    joblib.dump(gbm_model, f'{SAVE_DIR}{fault_name}_hgbm_model.pkl') # 추론용 HGBM 저장
    
    gbm_pred_train = np.expm1(gbm_model.predict(X_train)).clip(0, max_ttf)
    gbm_pred_aligned_full = np.expm1(gbm_model.predict(X_valid)).clip(0, max_ttf)

    # -------------------------------------
    # F. 딥러닝 시퀀스 생성 및 학습 & 💾 전처리기/모델 저장
    # -------------------------------------
    print(f"  [3] {model_type} 딥러닝 학습 및 저장 중...")
    preprocessor_dl = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_cols)
    ], remainder='drop')

    X_train_dl = preprocessor_dl.fit_transform(X_train)
    X_valid_dl = preprocessor_dl.transform(X_valid)
    joblib.dump(preprocessor_dl, f'{SAVE_DIR}{fault_name}_preprocessor_dl.pkl') # 추론용 전처리기 저장

    def create_aligned_sequences(X, y, ep_ids, gbm_preds):
        Xs, ys, g_preds, e_ids = [], [], [], []
        for ep in np.unique(ep_ids):
            idx = np.where(ep_ids == ep)[0]
            if len(idx) < SEQ_LEN: continue
            curr_X, curr_y, curr_g = X[idx], y[idx], gbm_preds[idx]
            for i in range(len(curr_X) - SEQ_LEN + 1):
                Xs.append(curr_X[i:i+SEQ_LEN])
                ys.append(curr_y[i+SEQ_LEN-1])
                g_preds.append(curr_g[i+SEQ_LEN-1])
                e_ids.append(ep)
        return np.array(Xs), np.array(ys), np.array(g_preds), np.array(e_ids)

    X_seq_train, y_seq_train, _, _ = create_aligned_sequences(X_train_dl, y_train_reg, train_part["episode_id"].values, gbm_pred_train)
    X_seq_valid, y_seq_valid, gbm_pred_aligned, ep_ids_valid = create_aligned_sequences(X_valid_dl, y_valid_capped, valid_part["episode_id"].values, gbm_pred_aligned_full)

    with tf.device('/CPU:0'):
        inputs = Input(shape=(X_seq_train.shape[1], X_seq_train.shape[2]))
        if model_type == "LSTM":
            x = LSTM(64, return_sequences=True)(inputs)
            x = LSTM(32)(x)
        else:
            x = GRU(64, return_sequences=True)(inputs)
            x = GRU(32)(x)

        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        dl_model = Model(inputs, outputs)

        dl_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        es = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        dl_model.fit(X_seq_train, y_seq_train, validation_split=0.2, epochs=20, batch_size=512, callbacks=[es], verbose=0)

        dl_model.save(f'{SAVE_DIR}{fault_name}_{model_type}_model.h5') # 추론용 딥러닝 모델 저장

        dl_pred_aligned = np.expm1(dl_model.predict(X_seq_valid, verbose=0).flatten()).clip(0, max_ttf)

    # -------------------------------------
    # G. 동적 앙상블 & Isotonic Regression & 💾 저장
    # -------------------------------------
    print("  [4] 최종 앙상블, Isotonic 교정 및 저장 중...")
    alpha = np.clip((max_ttf - dl_pred_aligned) / (max_ttf * 0.8), 0.1, 0.9)
    ensemble_pred = alpha * dl_pred_aligned + (1 - alpha) * gbm_pred_aligned

    final_iso_preds = np.zeros_like(ensemble_pred)
    iso_models_dict = {}
    
    for ep in np.unique(ep_ids_valid):
        idx = np.where(ep_ids_valid == ep)[0]
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
        final_iso_preds[idx] = iso.fit_transform(np.arange(len(idx)), ensemble_pred[idx])
        iso_models_dict[ep] = iso
        
    joblib.dump(iso_models_dict, f'{SAVE_DIR}{fault_name}_isotonic_dict.pkl') # Isotonic 후처리 저장

    # 평가
    rmse = np.sqrt(mean_squared_error(y_seq_valid, final_iso_preds))
    mae = mean_absolute_error(y_seq_valid, final_iso_preds)
    score = phm_score(y_seq_valid, final_iso_preds)

    print(f"  ✅ 완료! (RMSE: {rmse:,.0f} | MAE: {mae:,.0f} | Score: {score:,.0f} | 소요시간: {time.time() - start_time_all:.1f}초)")

    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "Fault": fault_name,
        "Downsample": f"{downsample_sec}초",
        "MAX_TTF": f"{max_ttf}초",
        "Model": f"{model_type} 앙상블",
        "RMSE": rmse,
        "MAE": mae,
        "PHM_Score": score
    }

# =========================================
# 5. 개별 조합 실행 (고정 파라미터 적용)
# =========================================
print("\n" + "🏁"*25)
print(" 🌟 다중 고장 모드(F1, F2) 챔피언 고정 파라미터 학습 및 저장 시작 🌟")
print("🏁"*25)

# F1, F2 루프
for fault_name, target_col in TARGET_DICT.items():
    print(f"\n\n⚙️  현재 타겟 고장 모드: {fault_name} ({target_col})")
    print(f"   => 최적화된 파라미터로 고정하여 학습 및 가중치 저장을 진행합니다.")
    
    p = FIXED_PARAMS[fault_name]
    res = run_pipeline(fault_name=fault_name, target_col=target_col, 
                       downsample_sec=p["rate"], max_ttf=p["ttf"], model_type=p["mod"])
    
    print(f"\n🏆 {fault_name} 고장 모드 최종 결과 🏆")
    print(f" 🥇 | 압축: {res['Downsample']} | 상한선: {res['MAX_TTF']} | 모델: {res['Model']} | RMSE: {res['RMSE']:,.0f} | MAE: {res['MAE']:,.0f} | 점수: {res['PHM_Score']:,.0f}\n")

print("\n" + "✅"*25)
print(f" 🌟 모든 고장 모드(F1, F2) 모델 학습 및 가중치 저장이 완료되었습니다! ({SAVE_DIR}) 🌟")
