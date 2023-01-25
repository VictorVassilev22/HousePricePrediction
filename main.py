from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.preprocessing as prep
from scipy.stats import norm
import statsmodels.api as sm
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from mlxtend.regressor import StackingCVRegressor
from joblib import dump
import warnings


# function for formatting columns
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


def formatter(df, istrain = True):
    columns_to_encode = list(df.select_dtypes(include=['category', 'object']))
    le = prep.LabelEncoder()
    for feature in columns_to_encode:
        try:
            if df[feature].dtype.name == 'category':
                df[feature] = df[feature].fillna("None", inplace=True)
            df[feature] = le.fit_transform(df[feature])
        except:
            print('Error encoding ' + feature)

    # "Utilities" can be deleted because it is equal to one value in almost all records.
    del df['Utilities']


    # LotFrontage (describes the junction of streets) will be grouped
    # by Neighborhood (the area in which the house is located),
    # and in each group, the empty LotFrontage values will be
    # replaced by the median of the group. This comes from the logic
    # that houses in the same area are roughly equally connected to the streets

    columns_to_encode = list(
        df.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']))
    for feature in columns_to_encode:
        df[feature] = df[feature].fillna(0)

    if istrain:
        train_df['LotFrontage'] = train_df['LotFrontage'].fillna(
            train_df.groupby('Neighborhood')['LotFrontage'].transform('mean'))
        train_df['MSZoning'].fillna(train_df['MSZoning'].mode()[0], inplace=True)
        train_df['Electrical'].fillna(train_df['Electrical'].mode()[0], inplace=True)
        train_df['KitchenQual'].fillna(train_df['KitchenQual'].mode()[0], inplace=True)
        train_df['Exterior1st'].fillna(train_df['Exterior1st'].mode()[0], inplace=True)
        train_df['Exterior2nd'].fillna(train_df['Exterior2nd'].mode()[0], inplace=True)
        train_df['SaleType'].fillna(train_df['SaleType'].mode()[0], inplace=True)

    return df


def reg_and_box(df, col_str):
    sns.regplot(x=df['SalePrice'], y=df[col_str])
    plt.show()
    sns.boxplot(x=df[col_str])
    plt.show()


# rmse calculates the root of mean squared error,
# and since the target variable is already at log space,
# this function calculates root mean squared log error which is the competition score metric.
def rmse(y_tr, y_pr):
    return np.sqrt(mean_squared_error(y_tr, y_pr))


def cv_rmse(model, x_tr, y_tr):
    return np.sqrt(-cross_val_score(model, x_tr, y_tr, scoring='neg_mean_squared_error', cv=kf))

def blend_predict(X):
    arr = ((0.05 * elastic_net.predict(X)) +
           (0.1 * lasso.predict(X)) +
           (0.05 * ridge.predict(X)) +
           (0.15 * gbr.predict(X)) +
           (0.15 * xgbr.predict(X)) +
           (0.1 * lgbmr.predict(X)) +
           (0.4 * stack.predict(X)))
    return arr


warnings.filterwarnings("ignore")
# 0. formatting data
# convert csv file to dataframe
train_df = pd.read_csv("train.csv")

# converting text variables with numerical values
train_df = formatter(train_df)

print(train_df)

# 1. Filtering data
corr = train_df.corr()
plt.figure(figsize=(45, 45))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, fmt='.2f')
plt.show()

sns.regplot(x=train_df['SalePrice'], y=train_df['GrLivArea'])
plt.show()
sns.regplot(x=train_df['SalePrice'], y=train_df['TotalBsmtSF'])
plt.show()
sns.boxplot(x=train_df['TotalBsmtSF'])
plt.show()
sns.boxplot(x=train_df['GrLivArea'])
plt.show()

# GrLivArea and TotalBsmtSF shows big correlation with the target but lots of outliers
# needs filtering
# doing iqr filtering to remove some outliers
percentile25_TBSF = train_df['TotalBsmtSF'].quantile(0.25)
percentile75_TBSF = train_df['TotalBsmtSF'].quantile(0.75)

percentile25_GLA = train_df['GrLivArea'].quantile(0.25)
percentile75_GLA = train_df['GrLivArea'].quantile(0.75)

q1_TBSF = train_df['TotalBsmtSF'].quantile(0.25)
q1_GLA = train_df['GrLivArea'].quantile(0.25)

q3_TBSF = train_df['TotalBsmtSF'].quantile(0.75)
q3_GLA = train_df['GrLivArea'].quantile(0.75)

iqr_TBSF = q3_TBSF - q1_TBSF
iqr_GLA = q3_GLA - q1_GLA

upper_limit_TBSF = percentile75_TBSF + 8 * iqr_TBSF
upper_limit_GLA = percentile75_GLA + 8 * iqr_GLA

lower_limit_TBSF = percentile25_TBSF - 1.5 * iqr_TBSF
lower_limit_GLA = percentile25_GLA - 1.5 * iqr_GLA

train_df = train_df[train_df['TotalBsmtSF'] < upper_limit_TBSF]
train_df = train_df[train_df['GrLivArea'] < upper_limit_GLA]

# this can be function
sns.regplot(x=train_df['SalePrice'], y=train_df['GrLivArea'])
plt.show()
sns.regplot(x=train_df['SalePrice'], y=train_df['TotalBsmtSF'])
plt.show()
sns.boxplot(x=train_df['TotalBsmtSF'])
plt.show()
sns.boxplot(x=train_df['GrLivArea'])
plt.show()

# Normalization of variables
spmean = train_df['SalePrice'].mean()
spsd = train_df['SalePrice'].std()
x = np.arange(0, 700000)
y = norm.pdf(x, loc=spmean, scale=spsd)
sns.distplot(train_df['SalePrice'], norm_hist=True, bins=60, kde=True, color='green').plot(x, y, 'purple')
plt.show()

sm.qqplot(train_df['SalePrice'], line='q')
plt.show()

train_df['SalePrice'] = train_df['SalePrice'].transform([np.log])
spmean = train_df['SalePrice'].mean()
spsd = train_df['SalePrice'].std()
x = np.arange(10, 14)
y = norm.pdf(x, loc=spmean, scale=spsd)
sns.distplot(train_df['SalePrice'], norm_hist=True, bins=40, kde=True, color='green')
sns.distplot(a=train_df['SalePrice'], bins=40, fit=norm, kde=False, color='red')
plt.show()

sm.qqplot(train_df['SalePrice'], line='q')
plt.show()

# prep
X_train = train_df.drop(columns=['SalePrice', 'Id']).values
y_train = train_df['SalePrice'].values

test_df = formatter(pd.read_csv("test.csv"), False)
X_test = test_df.drop(columns=['Id']).values

print('X_train shape: {}'.format(X_train.shape))
print('y_train shape: {}'.format(y_train.shape))
print('X_test shape: {}'.format(X_test.shape))

SEED = 42
K = 10
kf = KFold(n_splits=K, shuffle=True, random_state=SEED)

ridge = make_pipeline(RobustScaler(), KernelRidge(alpha=0.5))

lasso = make_pipeline(RobustScaler(), LassoCV(alphas=np.arange(0.0001, 0.0009, 0.0001), random_state=SEED, cv=kf))

elastic_net = make_pipeline(RobustScaler(), ElasticNetCV(alphas=np.arange(0.0001, 0.0008, 0.0001),
                                                         l1_ratio=np.arange(0.8, 1, 0.025), cv=kf))

gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=SEED)

xgbr = xgb.XGBRegressor(learning_rate=0.01, n_estimators=3500, max_depth=3,
                    gamma=0.001, subsample=0.7, colsample_bytree=0.7,
                    objective='reg:squarederror', nthread=-1, seed=SEED,
                    reg_alpha=0.0001)

lgbmr = lgb.LGBMRegressor(objective='regression',num_leaves=4, learning_rate=0.01,
                      n_estimators=5000, max_bin=200, bagging_fraction=0.75, bagging_freq=5,
                      bagging_seed=SEED, feature_fraction=0.2, feature_fraction_seed=SEED, verbose=-1)

stack = StackingCVRegressor(regressors=(ridge, lasso, elastic_net, gbr, lgbmr, xgbr), meta_regressor=xgbr, use_features_in_secondary=True)

# save models

# ridge = load('RidgeCV.joblib')
# lasso = load('LassoCV.joblib')
# elastic_net = load('ElasticNetCV.joblib')
# gbr = load('GradientBoostingRegressor.joblib')
# xgbr = load('XGBoostRegressor.joblib')
# lgbmr = load('LightGBMRegressor.joblib')
# stack = load('StackingCVRegressor.joblib')

models = {'RidgeCV': ridge,
          'LassoCV': lasso,
          'ElasticNetCV': elastic_net,
          'GradientBoostingRegressor': gbr,
          'LightGBMRegressor': lgbmr,
          'XGBoostRegressor': xgbr,
          'StackingCVRegressor': stack}
predictions = {}
scores = {}

for name, model in models.items():
    start = datetime.now()
    print('[{}] Running {}'.format(start, name))

    model.fit(X_train, y_train)
    predictions[name] = np.expm1(model.predict(X_train))

    score = cv_rmse(model, X_train, y_train)
    scores[name] = (score.mean(), score.std())

    end = datetime.now()

    print('[{}] Finished Running {} in {:.2f}s'.format(end, name, (end - start).total_seconds()))
    print('[{}] {} Mean RMSE: {:.6f} / Std: {:.6f}\n'.format(datetime.now(), name, scores[name][0], scores[name][1]))
    filename = name + ".joblib"
    dump(model, filename)


blended_score = rmse(y_train, blend_predict(X_train))
print('Blended Prediction RMSE: {}'.format(blended_score))

submission_df = pd.DataFrame(columns=['Id', 'SalePrice'])
submission_df['Id'] = test_df['Id']
submission_df['SalePrice'] = np.exp(blend_predict(X_test))
submission_df.to_csv('submissions.csv', header=True, index=False)
submission_df.head(10)
