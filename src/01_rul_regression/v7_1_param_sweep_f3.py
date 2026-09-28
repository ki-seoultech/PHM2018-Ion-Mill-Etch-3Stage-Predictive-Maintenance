"""
V7.1 — parameter sweep for F3 (downsample x MAX_TTF x LSTM/GRU)

- Adds diff / pct_change / mean-deviation / EMA / smoothed-diff features
- Ensemble weight alpha scales with MAX_TTF: clip((MAX_TTF - DL) / (0.8*MAX_TTF), 0.1, 0.9)
- NOTE: the script ranks by PHM Score only. The final F3 setting (140 s / 15,000 s / GRU)
  was chosen manually on the RMSE-MAE-PHM balance, not on this ranking (see README).

Input : ALL_M01_MASTER_labeled.csv
Output: sweep leaderboard (console)
"""

# =========================================================
# V7.1  V7 & Parameter Optimization (RMSE, MAE 출력 추가)
# =========================================================
import pandas as pd
import numpy as np
import time
import gc
import itertools
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
# 1. 환경 설정 및 공식 평가 함수
# =========================================
CSV_PATH = '/content/drive/MyDrive/ALL_M01_MASTER_labeled.csv'
TARGET_COL = "TTF_FlowCool Pressure Dropped Below Limit"
SEQ_LEN = 15

def phm_score(y_true, y_pred):
    return np.sum(np.exp(-0.001 * y_true) * np.abs(y_true - y_pred))

# =========================================
# 2. 대용량 데이터 로드 (불필요한 컬럼 원천 차단)
# =========================================
print(f"\n[INFO] 대용량 마스터 데이터 로드 중... (불필요한 컬럼 원천 차단)")
def load_filter(col):
    drop_keywords = ['Too High', 'leak', 'label', 'risk']
    for kw in drop_keywords:
        if kw in col: return False

    v6_drop_cols = ["ROTATIONSPEED", "Tool", "Lot", "runnum",
                    "ETCHAUX2SOURCETIMER", "ETCHSOURCEUSAGE",
                    "ETCHAUXSOURCETIMER", "ACTUALSTEPDURATION"]
    if col in v6_drop_cols:
        return False

    return True

df_raw = pd.read_csv(CSV_PATH, usecols=load_filter)

# 유효 구간 필터링 및 정답지 결측치 제거
if "FIXTURESHUTTERPOSITION" in df_raw.columns:
    df_raw = df_raw[df_raw['FIXTURESHUTTERPOSITION'] == 1].copy()
    df_raw.drop(columns=['FIXTURESHUTTERPOSITION'], inplace=True)

df_raw = df_raw.dropna(subset=[TARGET_COL]).copy()
df_raw.sort_values(['Unit', 'time'], inplace=True)

# 글로벌 에피소드 ID 생성 (장비가 바뀌거나 TTF가 오르면 새로운 고장 사이클로 인식)
df_raw['ttf_diff'] = df_raw[TARGET_COL].diff()
df_raw['is_new_ep'] = (df_raw['ttf_diff'] > 0) | (df_raw['Unit'] != df_raw['Unit'].shift())
df_raw['episode_id'] = df_raw['is_new_ep'].cumsum().astype(int)
df_raw.drop(columns=['ttf_diff', 'is_new_ep'], inplace=True)

print(f"       ✅ 로드 완료! (총 {len(df_raw):,} 행)")

# =========================================
# 3. V6.1 물리적 Feature Extraction 함수 보존
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
# 4. 핵심 파이프라인 (파라미터 주입 가능하도록 변수화)
# =========================================
def run_v7_pipeline(downsample_sec, max_ttf, model_type):
    print(f"\n" + "="*60)
    print(f" 🚀 [실험] 압축: {downsample_sec}초 | 상한선: {max_ttf}초 | 모델: {model_type}")
    print("="*60)
    start_time_all = time.time()

    # -------------------------------------
    # A. 데이터 다운샘플링
    # -------------------------------------
    df = df_raw.copy()
    df['time_bin'] = df['time'] // downsample_sec

    cat_cols = ['recipe', 'stage']
    ignore_cols = cat_cols + ['Unit', 'time', 'time_bin', 'episode_id', TARGET_COL, 'Tool', 'Lot', 'runnum', 'ROTATIONSPEED']
    num_cols = [c for c in df.columns if c not in ignore_cols and df[c].dtype != 'object']

    agg_dict = {'time': 'last', TARGET_COL: 'min'}
    for c in cat_cols:
        if c in df.columns: agg_dict[c] = 'last'
    for c in num_cols:
        agg_dict[c] = 'mean'

    df_down = df.groupby(['Unit', 'episode_id', 'time_bin']).agg(agg_dict).reset_index()
    print(f"  [1] 다운샘플링 완료: {len(df_down):,} 행 압축")

    # -------------------------------------
    # B. 물리적 Feature Extraction
    # -------------------------------------
    print("  [2] 물리적 Feature Extraction 진행 중...")
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
    # D. Unit-Aware Scaling
    # -------------------------------------
    print("  [3] 장비별(Unit) 센서 영점 정규화 진행 중...")
    drop_cols = ["time", "time_bin", "episode_id", "ROTATIONSPEED", "Tool", "Lot", "runnum", TARGET_COL]
    feature_cols = [c for c in train_part.columns if c not in drop_cols and c != 'Unit']
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

    X_train, y_train_raw = train_part[feature_cols], train_part[TARGET_COL].values
    X_valid, y_valid_raw = valid_part[feature_cols], valid_part[TARGET_COL].values

    y_train_capped = np.clip(y_train_raw, 0, max_ttf)
    y_valid_capped = np.clip(y_valid_raw, 0, max_ttf)
    y_train_reg = np.log1p(y_train_capped)

    # -------------------------------------
    # E. HGBM 모델 학습
    # -------------------------------------
    print("  [4] HGBM 모델 학습 중...")
    preprocessor_ml = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical_cols)
    ])

    gbm_model = Pipeline([
        ("prep", preprocessor_ml),
        ("model", HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_iter=250, random_state=42))
    ])

    gbm_model.fit(X_train, y_train_reg)
    gbm_pred_train = np.expm1(gbm_model.predict(X_train)).clip(0, max_ttf)
    gbm_pred_aligned_full = np.expm1(gbm_model.predict(X_valid)).clip(0, max_ttf)

    # -------------------------------------
    # F. LSTM/GRU 시퀀스 생성 및 학습
    # -------------------------------------
    print(f"  [5] {model_type} 전처리 및 Sequence 학습 중...")
    preprocessor_dl = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_cols)
    ], remainder='drop')

    X_train_dl = preprocessor_dl.fit_transform(X_train)
    X_valid_dl = preprocessor_dl.transform(X_valid)

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

    dl_pred_aligned = np.expm1(dl_model.predict(X_seq_valid, verbose=0).flatten()).clip(0, max_ttf)

    # -------------------------------------
    # G. 동적 앙상블 & Isotonic Regression
    # -------------------------------------
    print("  [6] 최종 앙상블 및 Isotonic 교정 중...")
    # 파라미터로 받은 max_ttf에 맞춰서 비율(alpha)이 자동 조절되도록 수식 수정
    alpha = np.clip((max_ttf - dl_pred_aligned) / (max_ttf * 0.8), 0.1, 0.9)
    ensemble_pred = alpha * dl_pred_aligned + (1 - alpha) * gbm_pred_aligned

    final_iso_preds = np.zeros_like(ensemble_pred)
    for ep in np.unique(ep_ids_valid):
        idx = np.where(ep_ids_valid == ep)[0]
        iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
        final_iso_preds[idx] = iso.fit_transform(np.arange(len(idx)), ensemble_pred[idx])

    # 평가
    rmse = np.sqrt(mean_squared_error(y_seq_valid, final_iso_preds))
    mae = mean_absolute_error(y_seq_valid, final_iso_preds)
    score = phm_score(y_seq_valid, final_iso_preds)

    print(f"  ✅ 완료! (RMSE: {rmse:,.0f} | MAE: {mae:,.0f} | Score: {score:,.0f} | 소요시간: {time.time() - start_time_all:.1f}초)")

    # 반복 실험 간 메모리 정리
    tf.keras.backend.clear_session()
    gc.collect()

    return {
        "Downsample": f"{downsample_sec}초",
        "MAX_TTF": f"{max_ttf}초",
        "Model": f"{model_type} 앙상블",
        "RMSE": rmse,
        "MAE": mae,
        "PHM_Score": score
    }

# =========================================
# 5. 개별 조합 실행 (V7 방식 그대로 유지)
# =========================================
rates = [120,140]
ttfs = [14000,15000,16000]
models = ["LSTM", "GRU"]

combinations = list(itertools.product(rates, ttfs, models))
results = []

print("\n" + "🏁"*20)
print(" 🌟 V7 기반 단일 파라미터 순차 대입 시작 🌟")
print("🏁"*20)

for rate, ttf, mod in combinations:
    res = run_v7_pipeline(downsample_sec=rate, max_ttf=ttf, model_type=mod)
    results.append(res)

print("\n" + "🏆"*25)
print(" 🌟 V7 개별 조합 스윕 최종 결과 🌟")
print("🏆"*25)

results_sorted = sorted(results, key=lambda x: x['PHM_Score'])

# 최종 리더보드 출력 (RMSE·MAE 포함)
for i, r in enumerate(results_sorted):
    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}위"
    print(f"{medal} | 압축: {r['Downsample']} | 상한선: {r['MAX_TTF']} | 모델: {r['Model']} | RMSE: {r['RMSE']:,.0f} | MAE: {r['MAE']:,.0f} | 점수: {r['PHM_Score']:,.0f}")

best_model = results_sorted[0]
print(f"\n🎉 최종 챔피언 세팅: {best_model['Downsample']}, {best_model['MAX_TTF']}, {best_model['Model']} (가장 낮은 페널티 달성!)")
