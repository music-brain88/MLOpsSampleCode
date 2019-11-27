
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix

import sklearn
import numpy as np
import pandas as pd
from PIL import Image
import logging
import os
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
    "max_depth": list(range(8, 12)),
    "n_estimators": list(range(1, 201, 50)),
    "bootstrap": [True, False],
    "max_features": list(range(5, 10))
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

model_cv.fit(x_train, y_trian)

model_cv_best = model_cv.best_estimator_
gird_param = dict(cv=5, scoring='neg_mean_squared_error')
print(gird_param)
parameters_dict = Task.current_task().connect(model_cv.best_params_)
Task.current_task().connect(gird_param)
print("Best Model Parameter: ", model_cv.best_params_)

# 予測を打ち込む
y_pred = model_cv.best_estimator_.predict(x_test)
print(y_pred)


joblib.dump(model_cv, 'model.pkl')
loaded_model = joblib.load('model.pkl')

number_layers = 10
accuracy = model_cv_best.score(x_test, y_test)

logger = Task.current_task().get_logger()
logger.report_scatter2d(
    "performance",
    "accuracy",
    iteration=0, 
    mode='markers',
    scatter=[
        (number_layers,
         accuracy
         )
    ]
)

keys = list(model_cv.cv_results_.keys())
for i in range(len(keys)):
    values = model_cv.cv_results_[keys[i]]
    key = keys[i]
    if isinstance(values[0], float):
        logger.report_histogram(
            key,
            key,
            iteration=0,
            values=values
        )
