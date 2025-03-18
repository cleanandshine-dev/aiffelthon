### 라이브러리 불러오기 ###
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# 언더 샘플링 및 오버 샘플링
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler  # 추가한 오버 샘플링 라이브러리

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


### Normal 데이터 샘플링 및 오버 샘플링 추가 시작 ###
normal_data = df[df['ff_sp_ai'] == 0]  # 정상 데이터
fraud_data = df[df['ff_sp_ai'] == 1]   # 사기 데이터

# 정상 데이터의 25% 비율로 샘플링
normal_data_sampled = normal_data.sample(frac=0.25, random_state=42)

# 오버샘플링
ros = RandomOverSampler(random_state=42)
X = pd.concat([normal_data_sampled[features], fraud_data[features]], axis=0)
y = pd.concat([normal_data_sampled[target], fraud_data[target]], axis=0)

X_resampled, y_resampled = ros.fit_resample(X, y)

# 데이터프레임 생성
df_resampled = pd.DataFrame(X_resampled, columns=features)
df_resampled[target] = y_resampled
### Normal 데이터 샘플링 및 오버 샘플링 추가 끝 ###

## 훈련데이터 및 검증데이터 확인
X_train = df_resampled[features]
y_train = df_resampled[target]

print("훈련데이터:", X_train.shape, y_train.shape)


### 머신러닝 모델링 ###
## LightGBM
lgbm = LGBMClassifier(random_state=42,
                     verbose=1,
                     max_depth=5,
                     n_estimators=500,
                     learning_rate=0.01,
                     num_leaves=5)

lgbm.fit(X_train, y_train,
         eval_set=[(X_val, y_val)])

lgbm_pred = lgbm.predict(X_test)
         
print(classification_report(y_test, lgbm_pred))


## Catboost
cbt = CatBoostClassifier(random_state=42,
                         iterations=500,
                         learning_rate=0.01,
                         verbose=-1,
                         cat_features=['wd_fc_ac', 'dps_fc_ac',
                                       'md_type', 'fnd_type'])

cbt.fit(X_train, y_train, eval_set=[(X_val, y_val)],
        use_best_model=True, verbose=True)

cbt_pred = cbt.predict(X_test)

print(classification_report(y_test, cbt_pred))


## XGBoost
xgb = XGBClassifier(n_estimators = 500,
                      learning_rate = 0.01,
                      objective ='binary:logistic',
                      scale_pos_weight = sum(y_train==0) / sum(y_train==1),
                      eval_metric='logloss',
                      enable_categorical = True,
                      early_stopping_rounds = 100,
                      seed=42)

xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])

xgb_pred = xgb.predict(X_test)
        
print(classification_report(y_test, xgb_pred))