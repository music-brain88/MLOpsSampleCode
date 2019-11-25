
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor


import sklearn
import numpy as np
import pandas as pd

# for trains
import joblib
from tqdm import tqdm

# task set
from trains import Task
task = Task.init(project_name='MLOps', task_name='sklearn random_forest')

# データの読み込み
boston = load_boston()

# 入出力の切り分け
x = boston['data']  # 物件の情報
y = boston['target']  # 家賃

model = RandomForestRegressor(n_jobs=1, random_state=2525)

param_grid = {
    "max_depth": list(range(1, 10)),
    "n_estimators": list(range(1, 500, 50)),
#    "bootstrap": [True, False],
    "max_features": list(range(1, 10))
}

model_cv = GridSearchCV(
    estimator=RandomForestRegressor(
        random_state=2525,
        verbose=True,
        n_jobs=6
    ),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=6
)

print(model_cv)

x_train, x_test, y_trian, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5, random_state=1)

tqdm(model_cv.fit(x_train, y_trian))

model_cv_best = model_cv.best_estimator_

parameters_dict = Task.current_task().connect(model_cv.best_params_)
print("Best Model Parameter: ", model_cv.best_params_)

#result = cross_validate(model, x_train, y_trian, cv=5)

joblib.dump(model_cv, 'model.pkl')
loaded_model = joblib.load('model.pkl')

# number_layers = 10
# accuracy = model.score(x_test, y_test)
# Task.current_task().get_logger().report_scatter2d(
#     "performance", "accuracy", iteration=0, 
#     mode='markers', scatter=[(number_layers, accuracy)])
