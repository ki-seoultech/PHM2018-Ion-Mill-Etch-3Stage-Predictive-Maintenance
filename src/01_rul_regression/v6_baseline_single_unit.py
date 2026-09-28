"""
V6 Baseline — single unit (01_M01), F3 (Pressure Dropped Below Limit)

- HGBM + LSTM baseline with MAX_TTF capping (50,000 s) and log1p target
- Idle-shutter removal, episode split by TTF reset, 10 physics-based window features
- Compares HGBM / LSTM / LSTM+Isotonic / dynamic ensemble / ensemble+Isotonic

Input : 01_M01_labeled_Final.csv (sensor data + TTF_* columns)
Output: RMSE / MAE / PHM Score per stage (console)
"""

# =========================================
# V6 Baseline Code (LSTM+Isotonic 추가 비교)
# =========================================
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression

# Colab이면 Drive 마운트
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

# =========================================
# 1. 데이터 로드 및 타겟 셋업
# =========================================
csv_path = '/content/drive/MyDrive/01_M01_labeled_Final.csv'
df = pd.read_csv(csv_path)

TARGET_COL = "TTF_FlowCool Pressure Dropped Below Limit"
df = df.dropna(subset=[TARGET_COL]).sort_values("time").reset_index(drop=True)

ttf_diff = df[TARGET_COL].diff()
df["episode_id"] = (ttf_diff > 0).fillna(False).cumsum().astype(int)

if "FIXTURESHUTTERPOSITION" in df.columns:
    df = df[df["FIXTURESHUTTERPOSITION"] == 1].copy()

# =========================================
# 2. 기계/물리적 Feature Extraction
# =========================================
def add_reference_features(df, sensor_cols, w=15):
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
    return df

core_sensors = ["FLOWCOOLPRESSURE", "IONGAUGEPRESSURE", "ETCHBEAMCURRENT", "FLOWCOOLFLOWRATE"]
core_sensors = [c for c in core_sensors if c in df.columns]

print("[INFO] Feature Extraction 진행 중...")
df = add_reference_features(df, core_sensors, w=15)

# =========================================
# 3. Train/Valid 분할 & Piecewise RUL
# =========================================
episodes = sorted(df["episode_id"].unique())
valid_eps = episodes[-2:]
train_part = df[~df["episode_id"].isin(valid_eps)].copy()
valid_part = df[df["episode_id"].isin(valid_eps)].copy()

drop_cols = ["time", "episode_id", "ROTATIONSPEED", "Tool", "Lot", "runnum",
             "ETCHAUX2SOURCETIMER", "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER", "ACTUALSTEPDURATION"]
drop_cols.extend([c for c in df.columns if "TTF" in c])
drop_cols.extend([c for c in df.columns if "label" in c.lower() or "risk" in c.lower()])

feature_cols = [c for c in train_part.columns if c not in drop_cols and train_part[c].nunique() > 1]
categorical_cols = [c for c in feature_cols if train_part[c].dtype == "object" or c in ["stage", "recipe"]]
numerical_cols = [c for c in feature_cols if c not in categorical_cols]

X_train, X_valid = train_part[feature_cols], valid_part[feature_cols]

MAX_TTF = 50000
y_train_capped = np.clip(train_part[TARGET_COL].values, 0, MAX_TTF)
y_valid_capped = np.clip(valid_part[TARGET_COL].values, 0, MAX_TTF)
y_train_reg = np.log1p(y_train_capped)

def phm_score(y_true, y_pred):
    return np.sum(np.exp(-0.001 * y_true) * np.abs(y_true - y_pred))

# =========================================
# 4. HGBM (머신러닝) 학습 및 예측
# =========================================
preprocessor_ml = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical_cols),
    ])

gbm_model = Pipeline([
    ("prep", preprocessor_ml),
    ("model", HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.05, max_iter=300, l2_regularization=0.1, random_state=42))
])

print("\n[INFO] 1. HistGradientBoosting 학습 중...")
gbm_model.fit(X_train, y_train_reg)
gbm_pred_train = np.expm1(gbm_model.predict(X_train)).clip(0, MAX_TTF)
gbm_pred_valid = np.expm1(gbm_model.predict(X_valid)).clip(0, MAX_TTF)

# =========================================
# 5. LSTM (딥러닝) 학습 및 예측
# =========================================
print("\n[INFO] 2. LSTM 전처리 및 학습 중...")
SEQ_LEN = 15

preprocessor_dl = ColumnTransformer(
    transformers=[("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numerical_cols)], remainder='drop')

X_train_dl = preprocessor_dl.fit_transform(X_train)
X_valid_dl = preprocessor_dl.transform(X_valid)

def create_aligned_sequences(X, y, ep_ids, gbm_preds, seq_len):
    Xs, ys, g_preds, e_ids = [], [], [], []
    for ep in np.unique(ep_ids):
        idx = np.where(ep_ids == ep)[0]
        curr_X, curr_y, curr_g = X[idx], y[idx], gbm_preds[idx]
        for i in range(len(curr_X) - seq_len + 1):
            Xs.append(curr_X[i:i+seq_len])
            ys.append(curr_y[i+seq_len-1])
            g_preds.append(curr_g[i+seq_len-1])
            e_ids.append(ep)
    return np.array(Xs), np.array(ys), np.array(g_preds), np.array(e_ids)

X_seq_train, y_seq_train, _, _ = create_aligned_sequences(X_train_dl, y_train_reg, train_part["episode_id"].values, gbm_pred_train, SEQ_LEN)
X_seq_valid, y_seq_valid, gbm_pred_aligned, ep_ids_valid = create_aligned_sequences(X_valid_dl, y_valid_capped, valid_part["episode_id"].values, gbm_pred_valid, SEQ_LEN)

inputs = Input(shape=(X_seq_train.shape[1], X_seq_train.shape[2]))
x = LSTM(64, return_sequences=True)(inputs)
x = LSTM(32)(x)
x = Dropout(0.2)(x)
x = Dense(16, activation='relu')(x)
outputs = Dense(1, activation='linear')(x)
lstm_model = Model(inputs, outputs)

lstm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit(X_seq_train, y_seq_train, validation_split=0.2, epochs=30, batch_size=256, callbacks=[es], verbose=0)

lstm_pred_aligned = np.expm1(lstm_model.predict(X_seq_valid, verbose=0).flatten()).clip(0, MAX_TTF)

# =========================================
# 6. 동적 앙상블 (Dynamic Blending)
# =========================================
alpha = np.clip((40000 - lstm_pred_aligned) / 30000, 0.1, 0.9)
ensemble_pred = alpha * lstm_pred_aligned + (1 - alpha) * gbm_pred_aligned

# =========================================
# 7. Isotonic Regression (단조 감소 교정)
# =========================================
print("[INFO] 3. Isotonic Regression (수명 단조 감소 교정) 적용 중...")
final_iso_preds = np.zeros_like(ensemble_pred)
lstm_iso_preds = np.zeros_like(lstm_pred_aligned) # 단일 LSTM용 교정 배열 추가

for ep in np.unique(ep_ids_valid):
    idx = np.where(ep_ids_valid == ep)[0]
    ep_preds = ensemble_pred[idx]
    lstm_ep_preds = lstm_pred_aligned[idx] # 단일 LSTM 예측값 추출

    iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
    x_time = np.arange(len(ep_preds))

    # 앙상블과 단일 LSTM 각각에 Isotonic 적용
    final_iso_preds[idx] = iso.fit_transform(x_time, ep_preds)
    lstm_iso_preds[idx] = iso.fit_transform(x_time, lstm_ep_preds)

# =========================================
# 8. 최종 결과 비교
# =========================================
results = {
    "1. HGBM (단일)": gbm_pred_aligned,
    "2. LSTM (단일)": lstm_pred_aligned,
    "3. LSTM (단일) + Isotonic": lstm_iso_preds,
    "4. 동적 앙상블": ensemble_pred,
    "5. 앙상블 + Isotonic (최종 비교)": final_iso_preds
}

print("\n================ 🏆 V6.1 모델 진화 단계별 성능 ================ ")
for name, preds in results.items():
    rmse = np.sqrt(mean_squared_error(y_seq_valid, preds))
    mae = mean_absolute_error(y_seq_valid, preds)
    score = phm_score(y_seq_valid, preds)

    print(f"[{name}]")
    print(f"  - RMSE      : {rmse:,.0f}")
    print(f"  - MAE       : {mae:,.0f}")
    print(f"  - PHM Score : {score:,.0f}\n")
