"""
Fault-cause classifier — final training and weight export (H = 14,000 s)

- Multi-class SMOTE + XGBoost, validation Macro F1 printed

Input : ALL_M01_MASTER_labeled_V2.csv
Output: classifier_preprocessor.pkl, champion_classifier.pkl
"""

# =========================================================
# 고장원인(F1,F2,F3) 다중분류기 최종 학습 및 가중치 저장
# =========================================================
import pandas as pd
import numpy as np
import time
import gc
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

# =========================================
# 1. 환경 설정 및 고장 모드 정의
# =========================================
CSV_PATH = '/content/drive/MyDrive/ALL_M01_MASTER_labeled_V2.csv'
SAVE_DIR = '/content/drive/MyDrive/PdM_Models_Regression/' 
os.makedirs(SAVE_DIR, exist_ok=True)

TARGET_DICT = {
    0: "TTF_Flowcool Pressure Too High Check Flowcool Pump", 
    1: "TTF_Flowcool leak",                                  
    2: "TTF_FlowCool Pressure Dropped Below Limit"           
}
target_cols = list(TARGET_DICT.values())

HIGH_THRESH = 14000 
LOW_THRESH = 50000
DOWNSAMPLE_SEC = 140

# =========================================
# 2. 데이터 로드 및 피처 엔지니어링
# =========================================
print(f"\n[INFO] 대용량 마스터 데이터 로드 중 (V2 데이터 장착)...")
def load_filter(col):
    if 'label' in col.lower() or 'risk' in col.lower() or col in ['Tool', 'Lot']: return False
    return True

df_raw = pd.read_csv(CSV_PATH, usecols=load_filter)
if "FIXTURESHUTTERPOSITION" in df_raw.columns:
    df_raw = df_raw[df_raw['FIXTURESHUTTERPOSITION'] == 1].copy()
    df_raw.drop(columns=['FIXTURESHUTTERPOSITION'], inplace=True)

df_raw['min_TTF'] = df_raw[target_cols].min(axis=1)
df_raw['Fault_Type'] = df_raw[target_cols].values.argmin(axis=1)

df_raw = df_raw.dropna(subset=['min_TTF']).copy()
df_raw.sort_values(['Unit', 'time'], inplace=True)

df_raw['ttf_diff'] = df_raw['min_TTF'].diff()
df_raw['is_new_ep'] = (df_raw['ttf_diff'] > 0) | (df_raw['Unit'] != df_raw['Unit'].shift())
df_raw['episode_id'] = df_raw['is_new_ep'].cumsum().astype(int)
df_raw.drop(columns=['ttf_diff', 'is_new_ep'], inplace=True)

df = df_raw.copy()
df['time_bin'] = df['time'] // DOWNSAMPLE_SEC

cat_cols = ['recipe', 'stage']
ignore_cols = cat_cols + ['Unit', 'time', 'time_bin', 'episode_id', 'min_TTF', 'Fault_Type'] + target_cols
num_cols = [c for c in df.columns if c not in ignore_cols and df[c].dtype != 'object']

agg_dict = {'time': 'last', 'min_TTF': 'min', 'Fault_Type': 'last'}
for c in target_cols: agg_dict[c] = 'min'
for c in cat_cols:
    if c in df.columns: agg_dict[c] = 'last'
for c in num_cols:
    agg_dict[c] = 'mean'

df_down = df.groupby(['Unit', 'episode_id', 'time_bin']).agg(agg_dict).reset_index()

def add_reference_features(df, sensor_cols, w=10):
    df = df.copy()
    g = df.groupby("episode_id", sort=False)
    for col in sensor_cols:
        roll = g[col].rolling(window=w, min_periods=2)
        df[f"{col}_mean"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{col}_std"] = roll.std().reset_index(level=0, drop=True)
        df[f"{col}_max"] = roll.max().reset_index(level=0, drop=True)
        df[f"{col}_rms"] = roll.apply(lambda x: np.sqrt(np.mean(np.square(x))), raw=True).reset_index(level=0, drop=True)
        df[f"{col}_diff"] = g[col].diff()
        df[f"{col}_mean_diff"] = df[col] - df[f"{col}_mean"]
        df[f"{col}_ema"] = g[col].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

core_sensors = ["FLOWCOOLPRESSURE", "IONGAUGEPRESSURE", "ETCHBEAMCURRENT", "FLOWCOOLFLOWRATE"]
core_sensors = [c for c in core_sensors if c in df_raw.columns]

print("[INFO] 피처 엔지니어링 진행 중...")
df_down = add_reference_features(df_down, core_sensors, w=10)
df_down.bfill(inplace=True)

# Train/Valid 분할
episodes_per_unit = df_down.groupby('Unit')['episode_id'].unique()
valid_eps = [eps[-1] for eps in episodes_per_unit if len(eps) > 1]
train_part = df_down[~df_down['episode_id'].isin(valid_eps)].copy()
valid_part = df_down[df_down['episode_id'].isin(valid_eps)].copy()

feature_cols = [c for c in train_part.columns if c not in ['Unit', 'time', 'time_bin', 'episode_id', 'min_TTF', 'Fault_Type'] and not c.startswith('TTF_')]
categorical_cols = [c for c in feature_cols if train_part[c].dtype.name == "category" or c in cat_cols]
numerical_cols = [c for c in feature_cols if c not in categorical_cols]

# =========================================
# 3. 모델 학습 및 💾 가중치 영구 저장
# =========================================
print(f"\n" + "="*60)
print(f" 🚀 [다중 분류기] 챔피언 고정(14000초) 학습 및 저장")
print("="*60)

# H(위험) 구간의 데이터만 가지고 학습합니다.
train_mc = train_part[train_part['min_TTF'] <= HIGH_THRESH].copy()
valid_mc = valid_part[valid_part['min_TTF'] <= HIGH_THRESH].copy()

X_train_mc, y_train_mc = train_mc[feature_cols], train_mc['Fault_Type'].values
X_valid_mc, y_valid_mc = valid_mc[feature_cols], valid_mc['Fault_Type'].values

# 💾 분류기 전용 전처리기 학습 및 저장
preprocessor = ColumnTransformer(transformers=[
    ("num", SimpleImputer(strategy="median"), numerical_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ]), categorical_cols)
])

X_train_mc_prep = preprocessor.fit_transform(X_train_mc)
X_valid_mc_prep = preprocessor.transform(X_valid_mc) # 검증용 전처리

joblib.dump(preprocessor, f'{SAVE_DIR}classifier_preprocessor.pkl') # 전처리기 저장

print("  [1] 고장 원인 다중 분류기 학습 중 (Multi-Class SMOTE 적용)...")
mc_smote = SMOTE(random_state=42)
X_train_mc_bal, y_train_mc_bal = mc_smote.fit_resample(X_train_mc_prep, y_train_mc)

xgb_model = XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                          max_depth=6, learning_rate=0.1, n_estimators=150, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_mc_bal, y_train_mc_bal)

# 💡 검증 데이터로 예측 후 성능 출력!
pred_mc = xgb_model.predict(X_valid_mc_prep)
f1_mc = f1_score(y_valid_mc, pred_mc, average='macro')
print(f"  📊 결과 확인 | 원인분류 Macro F1-Score: {f1_mc:.4f}")

# 💾 다중 분류기 챔피언 가중치 저장
joblib.dump(xgb_model, f'{SAVE_DIR}champion_classifier.pkl')

print(f"\n🎉 완료! 분류기 가중치(preprocessor, xgb_model)가 드라이브에 영구 저장되었습니다.")
