### 라이브러리 불러오기 ###
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# 언더 샘플링 #
from imblearn.under_sampling import RandomUnderSampler  # 랜덤 언더샘플링
from imblearn.under_sampling import TomekLinks          # Tomek Links
from imblearn.under_sampling import CondensedNearestNeighbour    # CNN(Condensed Nearest Neighbors)
from imblearn.under_sampling import NearMiss            # NearMiss
from imblearn.under_sampling import ClusterCentroids    # ClusterCentroids


# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler # 데이터 스케일링
from sklearn.preprocessing import LabelEncoder # 데이터 인코딩
from zoneinfo import ZoneInfo # 시간 인코딩

# 머신러닝 모델
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 머신러닝 모델 평가 지표
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix

# 하이퍼 파라미터 튜닝
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV



### 데이터 불러오기 ###
df_new = pd.read_csv(r'C:\Users\DATOP\data\dataset_002\HF_TRANS_TRAN_new.csv')
df_new.info()


## 데이터 복사 ##
df = df_new.copy()


### 데이터 결측치 확인 ###
df.isna().sum()

df['ff_sp_ai'].value_counts(dropna=False)

## 데이터 결측치 채우기 ##
df['ff_sp_ai'].fillna(0, inplace=True)
df.isna().sum()

## ff_sp_ai 컬럼 맵핑 - 정상거래 0, 의심거래 1
label_mapping = {0: 0, '01': 0, '02': 0, 1: 0, 2: 0, 'SP': 1}
df['ff_sp_ai'] = df['ff_sp_ai'].map(label_mapping)

df['ff_sp_ai'].value_counts()
df['ff_sp_ai'].dtype



### 데이터 타입 변환 ###
# 타입 변환 전
df.info()

# tran_dt : int64 -> datetime
df['tran_dt'] = pd.to_datetime(df['tran_dt'], format='%Y%m%d')

# wd_fc_ac, dps_fc_ac, md_type, fnd_type : int64 -> category
df['wd_fc_ac'] = df['wd_fc_ac'].astype('category')
df['dps_fc_ac'] = df['dps_fc_ac'].astype('category')
df['md_type'] = df['md_type'].astype('category')
df['fnd_type'] = df['fnd_type'].astype('category')

# 타입변환 후 
df.info()



### 날짜/시간 컬럼 ###
# 날짜데이터 분리
df['month'] = df['tran_dt'].dt.month
df['day'] = df['tran_dt'].dt.day
df['weekday'] = df['tran_dt'].dt.weekday  # 0: 월요일 ~ 6: 일요일

# 주기성을 갖는 sin,cos 컬럼 생성
df['hour_sin'] = round(np.sin(2 * np.pi * df['tran_tmrg'] / 24), 4)
df['hour_cos'] = round(np.cos(2 * np.pi * df['tran_tmrg'] / 24), 4)
df['weekday_sin'] = round(np.sin(2 * np.pi * df['weekday'] / 7), 4)
df['weekday_cos'] = round(np.cos(2 * np.pi * df['weekday'] / 7), 4)
df.head()



### 훈련, 검증, 시험 데이터로 나누기 ###
# 컬럼 분리
features = ['wd_fc_ac', 'dps_fc_ac', 'md_type', 'fnd_type',
            'tran_amt', 'month', 'day', 'hour_sin', 'hour_cos',
            'weekday_sin', 'weekday_cos']
target = 'ff_sp_ai'


## 데이터 나누기
X_train = df.loc[(df['month'] >= 1) & (df['month'] < 10), features]  # 1월 ~ 9월
X_val = df.loc[df['month'] == 10, features]                          # 10월 
X_test = df.loc[(df['month'] >= 11) & (df['month'] <= 12), features]  # 11~12월

y_train = df.loc[(df['month'] >= 1) & (df['month'] < 10), target]    # 1월 ~ 9월
y_val = df.loc[df['month'] == 10, target]                            # 10월
y_test = df.loc[(df['month'] >= 11) & (df['month'] <= 12), target]   # 11~12월

print("훈련데이터:", X_train.shape, y_train.shape)
print("검증데이터:", X_val.shape, y_val.shape)
print("시험데이터:", X_test.shape, y_test.shape)


## 훈련데이터 확인
display(X_train.head(3))
display(X_train.tail(3))
y_train.value_counts(normalize=True)


### 언더샘플링 - RandomUnderSampling ###
# 비율 설정 (0.99,...,0.90)
ratios = np.arange(0.99, 0.89, -0.01)

# 샘플링 결과를 저장할 딕셔너리
sampled_datasets = {}

# 의심거래개수
num_minority = np.sum(y_train == 1)

for ratio in ratios:
    # 정상거래의 목표 개수 설정
    num_majority = int(num_minority / (1 - ratio) - num_minority)

    # RandomUnderSampler 적용
    rus = RandomUnderSampler(sampling_strategy={0: num_majority, 1: num_minority}, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    # 저장
    sampled_datasets[ratio] = (X_resampled, y_resampled)

    # 데이터 개수 확인
    print(f"비율 {ratio:.1f}: 0 클래스 {num_majority}, 1 클래스 {num_minority}")

# 샘플링된 데이터셋을 사용해 모델을 학습할 수 있음


### 머신러닝 모델링 ###
## LightGBM
# 평가 지표를 저장할 리스트
results = []

# 하이퍼파라미터 설정 (기본값, 필요시 변경 가능)
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 5,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'verbose': -1,
    'max_depth':5,
    'random_state':42
}

# 각 비율별로 LightGBM 모델 학습 및 평가
for ratio, (X_train_resampled, y_train_resampled) in sampled_datasets.items():
    print(f"\n🔹 정상거래비율 {ratio:.2f}로 LightGBM 학습 중...")

    # LightGBM 모델 생성 및 학습
    model = lgb.LGBMClassifier(**lgb_params)
    model.fit(X_train_resampled, y_train_resampled, eval_set=[(X_val, y_val)])

    # 예측 수행
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Positive class 확률

    # 평가 지표 계산
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    # 결과 저장
    results.append({
        "비율": ratio, "Accuracy": acc, "Precision": prec, "Recall": rec, 
        "F1-score": f1, "ROC-AUC": roc_auc
    })


# 결과 정리
df_results = pd.DataFrame(results)

# 성능 비교 출력
print("\n📊 LightGBM성능 비교 결과")
print(df_results)

# 성능 결과를 CSV 파일로 저장 (옵션)
df_results.to_csv("random_lightgbm_results.csv", index=False)


## Catboost
# 평가 지표를 저장할 리스트
results = []

# CatBoostClassifier 하이퍼파라미터 설정
cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'iterations': 500,
    'learning_rate': 0.05,
    'depth': 6,
    'verbose': 0,
    'early_stopping_rounds': 50,
    'cat_features':['wd_fc_ac', 'dps_fc_ac','md_type', 'fnd_type']
}

# 각 비율별로 CatBoost 모델 학습 및 평가
for ratio, (X_train_resampled, y_train_resampled) in sampled_datasets.items():
    print(f"\n🔹 정상거래비율 {ratio:.2f}로 CatBoostClassifier 학습 중...")

    # CatBoost 모델 생성 및 학습
    model = cb.CatBoostClassifier(**cat_params)
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    # 예측 수행
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Positive class 확률

    # 평가 지표 계산
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    # 결과 저장
    results.append({
        "비율": ratio, "Accuracy": acc, "Precision": prec, "Recall": rec, 
        "F1-score": f1, "ROC-AUC": roc_auc
    })

# 결과 정리
df_results = pd.DataFrame(results)

# 성능 비교 출력
print("\n📊 CatBoost 성능 비교 결과")
print(df_results)

# 성능 결과를 CSV 파일로 저장 (옵션)
df_results.to_csv("random_catboost_results.csv", index=False)


## XGBoost
# 평가 지표를 저장할 리스트
results = []

# XGBoost 하이퍼파라미터 설정 (기본값, 필요시 변경 가능)
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 500,
    'early_stopping_rounds': 50,
    'scale_pos_weight': sum(y_train==0) / sum(y_train==1),
    'enable_categorical':True,
    'seed':42
}

# 각 비율별로 XGBoost 모델 학습 및 평가
for ratio, (X_train_resampled, y_train_resampled) in sampled_datasets.items():
    print(f"\n🔹 정상거래비율 {ratio:.2f}로 XGBoostClassifier 학습 중...")

    # XGBoost 모델 생성
    model = xgb.XGBClassifier(**xgb_params)

    # 학습 진행 (검증 데이터 포함)
    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_val, y_val)]
    )

    # 예측 수행
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Positive class 확률

    # 평가 지표 계산
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    

    # 결과 저장
    results.append({
        "비율": ratio, "Accuracy": acc, "Precision": prec, "Recall": rec, 
        "F1-score": f1, "ROC-AUC": roc_auc
    })

# 결과 정리
df_results = pd.DataFrame(results)

# 성능 비교 출력
print("\n📊 XGBoost 성능 비교 결과")
print(df_results)

# 성능 결과를 CSV 파일로 저장 (옵션)
df_results.to_csv("random_xgboost_results.csv", index=False)
