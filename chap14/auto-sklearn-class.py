import numpy as np
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# 数据部分

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 异或训练集
y_train = np.array([0, 1, 1, 0])

X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 异或训练集
y_test = np.array([0, 1, 1, 0])

# 自动训练部分
automl = autosklearn.classification.AutoSklearnClassifier()  # 创建自动分类器
automl.fit(X_train, y_train)  # 自动建模，自动调参
y_hat = automl.predict(X_test)  # 预测
print("准确率分数：", sklearn.metrics.accuracy_score(y_test, y_hat))
print(automl.cv_results_)
print(automl.sprint_statistics())
print(automl.show_models())  # 打印适配的模型的信息
