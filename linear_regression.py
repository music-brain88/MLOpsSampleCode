from sklearn.datasets import load_boston
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# task set
from trains import Task
task = Task.init(project_name='MLOps', task_name='sklearn regression')

# データの読み込み
boston = load_boston()


# 入出力の切り分け
x = boston['data']  # 物件の情報
y = boston['target']  # 家賃


model = LinearRegression()

print(model.fit(x, y))

print(model.score(x, y))


x_train, x_test, y_trian, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5, random_state=1)

model2 = LinearRegression()

print(model2.fit(x_train, y_trian))

print(model2.score(x_train, y_trian))

print(model2.score(x_test, y_test))
