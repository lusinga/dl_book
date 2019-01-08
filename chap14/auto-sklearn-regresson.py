import numpy as np
import autosklearn.classification
import autosklearn.regression
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
# 数据部分
X_train = np.arange(1, 10001)
y_train = X_train * 2 + np.random.random(10000)  # 生成2 * X + (0,1)之间的随机数
X_test = np.arange(1, 101)
y_test = X_test * 2
X_train = X_train.reshape(-1,
                          1)  # 按照auto-sklearn的要求，不能是一维向量，一维的也要reshape成(-1,1)型的
X_test = X_test.reshape(-1, 1)
# 训练部分
feature_types = (['numerical'] * 1)  # 只有一个数字型的feature
automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=120,
    per_run_time_limit=30,
    tmp_folder='~/tmp/autosklearn_regression_example_tmp',
    output_folder='~/tmp/autosklearn_regression_example_out',
)
automl.fit(X_train, y_train, feat_type=feature_types)  # 自动训练模型
print(automl.show_models())  # 打印模型信息
predictions = automl.predict(X_test)  #根据自动生成的
print("R2分数:", sklearn.metrics.r2_score(y_test, predictions))  # 评测分数
