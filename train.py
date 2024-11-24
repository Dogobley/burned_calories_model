import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import pickle


# Route variables
output_file = 'deploy/model.bin'
df = pd.read_csv('dataset.csv')

df.columns = df.columns.str.lower().str.replace(' ','_')

def vectorizer(df, features):
    train_dicts = df[features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(train_dicts)

    return dv, X

# Split dataset

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train['calories_burned']
y_val = df_val['calories_burned']
y_test = df_test['calories_burned']

del df_train['calories_burned']
del df_val['calories_burned']
del df_test['calories_burned']

# Transforming categorized data

dv, X_train = vectorizer(df_train, df_train.columns)
_, X_val = vectorizer(df_val, df_val.columns)
_, X_test = vectorizer(df_test, df_test.columns)

# Defining basic params and dict of iterators to find the best model

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'seed': 1
}

param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.3]
}

xgb_reg = XGBRegressor(objective='reg:squarederror', seed=1)


random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=param_grid,
    n_iter=50,
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Best XGBOOST params

best_params = random_search.best_params_

if 'n_estimators' in best_params:
    num_boost_round = best_params.pop('n_estimators')
else:
    num_boost_round = 100

params.update(best_params)

print("best params are: ", params, "\n", "Number of boost rounds are: ", num_boost_round)

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dv.get_feature_names_out().tolist())
dval = xgb.DMatrix(X_val, label=y_val, feature_names=dv.get_feature_names_out().tolist())
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=dv.get_feature_names_out().tolist())

dt = xgb.train(params, dtrain, num_boost_round)

y_pred_val = dt.predict(dval)
print("val", root_mean_squared_error(y_val, y_pred_val))

y_pred_test = dt.predict(dtest)
print("test", root_mean_squared_error(y_test, y_pred_test))

with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, dt), f_out)

print(f'the model is saved to {output_file}')