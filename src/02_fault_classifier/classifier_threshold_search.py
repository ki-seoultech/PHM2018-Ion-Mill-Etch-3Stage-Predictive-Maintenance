"""
Fault-cause classifier — danger-label threshold search

- H label = min TTF <= {5,000 / 10,000 / 14,000} s, L label = min TTF >= 50,000 s
- L/H binary: SMOTE + undersampling + RandomForest
- Fault cause (F1/F2/F3): XGBoost multi:softprob, Macro F1
- Threshold ranked by total score (L/H F1 + cause F1) -> 14,000 s selected

Input : ALL_M01_MASTER_labeled_V2.csv
Output: score table per threshold (console)
"""

# =========================================================
# 고장원인(F1,F2,F3) 다중분류기 Parameter Optimization
# =========================================================
import pandas as pd
import numpy as np
import time
import gc

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

# imbalanced-learn 라이브러리 사용
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Colab Drive Mount
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

# =========================================
# 1. 환경 설정 및 고장 모드 정의
# =========================================
# [변경 1] V2 마스터 데이터 사용 (노후화 지표·공정 문맥 포함)
CSV_PATH = '/content/drive/MyDrive/ALL_M01_MASTER_labeled_V2.csv'

TARGET_DICT = {
    "F1": "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "F2": "TTF_Flowcool leak",
    "F3": "TTF_FlowCool Pressure Dropped Below Limit"
}
target_cols = list(TARGET_DICT.values())

LOW_THRESH = 50000
HIGH_THRESH_CANDIDATES = [5000, 10000, 14000]


DOWNSAMPLE_SEC = 140

# =========================================
# 2. 데이터 로드 및 피처 엔지니어링
# =========================================
print(f"\n[INFO] 대용량 마스터 데이터 로드 중 (V2 데이터 장착)...")
def load_filter(col):
    if 'label' in col.lower() or 'risk' in col.lower() or col in ['Tool', 'Lot']: return False
    # [변경 2] 노후화 지표(runnum 등)는 삭제하지 않고 유지 (기존 삭제 코드는 주석 처리)
    # v6_drop_cols = ["ROTATIONSPEED", "runnum", "ETCHAUX2SOURCETIMER", "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER", "ACTUALSTEPDURATION"]
    # if col in v6_drop_cols: return False
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

# [변경 3] 공정 문맥 변수(recipe, stage) 유지
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

# 물리 피처는 V8 기준의 경량 버전 사용
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
# 3. 분류 모델 최적화 실험 (Grid Search)
# =========================================
def run_classification_experiment(high_thresh):
    print(f"\n" + "="*60)
    print(f" 🚀 [실험] 위험(H) 라벨링 기준: TTF <= {high_thresh}초")
    print("="*60)

    train_df = train_part.copy()
    valid_df = valid_part.copy()

    # Label: 1 (High Risk), 0 (Low Risk), NaN (Buffer)
    train_df['LH_Label'] = np.where(train_df['min_TTF'] <= high_thresh, 1,
                                   np.where(train_df['min_TTF'] >= LOW_THRESH, 0, np.nan))
    valid_df['LH_Label'] = np.where(valid_df['min_TTF'] <= high_thresh, 1,
                                   np.where(valid_df['min_TTF'] >= LOW_THRESH, 0, np.nan))

    train_lh = train_df.dropna(subset=['LH_Label']).copy()
    valid_lh = valid_df.dropna(subset=['LH_Label']).copy()

    X_train_lh, y_train_lh = train_lh[feature_cols], train_lh['LH_Label'].values
    X_valid_lh, y_valid_lh = valid_lh[feature_cols], valid_lh['LH_Label'].values

    # [변경 4] 공정 문맥 변수 인코딩용 전처리기
    preprocessor = ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ]), categorical_cols)
    ])

    X_train_lh_prep = preprocessor.fit_transform(X_train_lh)
    X_valid_lh_prep = preprocessor.transform(X_valid_lh)

    print("  [1] L/H 이진 분류기 학습 중 (SMOTE 적용)...")
    smote_pipeline = ImbPipeline([
        ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),
        ('under', RandomUnderSampler(sampling_strategy=0.8, random_state=42)),
        ('model', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])

    smote_pipeline.fit(X_train_lh_prep, y_train_lh)
    pred_lh = smote_pipeline.predict(X_valid_lh_prep)
    f1_lh = f1_score(y_valid_lh, pred_lh)

    # ------------------------------------------------
    # [과제 2] 고장 원인 다중 분류기
    # ------------------------------------------------
    train_mc = train_df[train_df['min_TTF'] <= high_thresh].copy()
    valid_mc = valid_df[valid_df['min_TTF'] <= high_thresh].copy()

    X_train_mc, y_train_mc = train_mc[feature_cols], train_mc['Fault_Type'].values
    X_valid_mc, y_valid_mc = valid_mc[feature_cols], valid_mc['Fault_Type'].values

    X_train_mc_prep = preprocessor.fit_transform(X_train_mc)
    X_valid_mc_prep = preprocessor.transform(X_valid_mc)

    print("  [2] 고장 원인 다중 분류기 학습 중 (XGBoost)...")
    xgb_model = XGBClassifier(objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                              max_depth=6, learning_rate=0.1, n_estimators=150, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_mc_prep, y_train_mc)
    pred_mc = xgb_model.predict(X_valid_mc_prep)

    f1_mc = f1_score(y_valid_mc, pred_mc, average='macro')

    print(f"  ✅ 완료! | L/H F1-Score: {f1_lh:.4f} | 원인분류 F1-Score: {f1_mc:.4f}")

    return {
        "High_Thresh": high_thresh,
        "LH_F1_Score": f1_lh,
        "MultiClass_F1_Score": f1_mc,
        "Total_Score": f1_lh + f1_mc
    }

# =========================================
# 4. 루프 실행 및 최적 기준점 도출
# =========================================
print("\n" + "🏁"*20)
print(" 🌟 V2 데이터 적용 분류기 성능 테스트 시작 🌟")
print("🏁"*20)

results = []
for h_thresh in HIGH_THRESH_CANDIDATES:
    res = run_classification_experiment(h_thresh)
    results.append(res)

print("\n" + "🏆"*20)
print(" 🌟 분류기 스윕 최적화 최종 결과 🌟")
print("🏆"*20)

results_sorted = sorted(results, key=lambda x: x['Total_Score'], reverse=True)

for i, r in enumerate(results_sorted):
    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}위"
    print(f"{medal} | H 라벨링 기준: TTF <= {r['High_Thresh']}초 | L/H F1: {r['LH_F1_Score']:.4f} | 원인분류 F1: {r['MultiClass_F1_Score']:.4f}")
