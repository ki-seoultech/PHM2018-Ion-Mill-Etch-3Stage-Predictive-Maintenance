"""
Build the classifier dataset (V2)

- Merges per-unit labeled CSVs and keeps aging counters (runnum, ETCHSOURCEUSAGE, ...)
  and process context (recipe, stage)
- Unit ID = unit + Lot (e.g. 01_M01_Lot1) so each maintenance cycle is an independent series

Input : *_M01_labeled_Final.csv
Output: ALL_M01_MASTER_labeled_V2.csv
"""

###ALL_M01_MASTER_labeled_V2###
import pandas as pd
import glob
import os

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception:
    pass

FOLDER_PATH = '/content/drive/MyDrive/'
SAVE_PATH = '/content/drive/MyDrive/ALL_M01_MASTER_labeled_V2.csv'

search_pattern = os.path.join(FOLDER_PATH, "*_M01_labeled_Final.csv")
all_files = sorted(glob.glob(search_pattern))

if not all_files:
    print("❌ 파일을 찾을 수 없습니다. 경로를 확인해 주세요.")
else:
    df_list = []
    print(f"총 {len(all_files)}개의 파일 병합을 시작합니다...\n")

    for file in all_files:
        filename = os.path.basename(file)
        print(f"[{filename}] 로드 및 정제 중...")
        df = pd.read_csv(file)

        # 시계열 독립성 유지: Unit 이름에 Lot 번호 결합
        unit_name = filename.split('_labeled')[0]
        if 'Lot' in df.columns:
            # ex) "01_M01_Lot1", "01_M01_Lot2" 처럼 기계 가동 주기별로 고유 이름표 생성
            df['Unit'] = unit_name + "_Lot" + df['Lot'].astype(str)
        else:
            df['Unit'] = unit_name

        # 1. 기존 라벨 컬럼 삭제
        cols_to_drop = [c for c in df.columns if 'label' in c.lower() or 'risk' in c.lower()]

        # 2. Tool, Lot 삭제 (Lot 정보는 Unit 이름에 반영됨)
        cols_to_drop.extend(['Tool', 'Lot'])

        cols_to_drop = list(set(cols_to_drop).intersection(df.columns))
        df.drop(columns=cols_to_drop, inplace=True)

        df_list.append(df)

    print("\n모든 파일 병합 중...")
    master_df = pd.concat(df_list, ignore_index=True)

    print("V2 마스터 CSV 파일로 덮어쓰기 저장 중...")
    master_df.to_csv(SAVE_PATH, index=False)
    print(f"✅ V2.1 마스터 데이터 생성 완료! (저장 경로: {SAVE_PATH}, 총 데이터 수: {len(master_df):,})")
