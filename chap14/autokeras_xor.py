import numpy as np
from autokeras import MlpModule
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 异或训练集
y_train = np.array([0, 1, 1, 0])

X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 异或训练集
y_test = np.array([0, 1, 1, 0])

mlpModule = MlpModule(
    loss=classification_loss, metric=Accuracy, searcher_args={},
    verbose=True)  # 多层神经网络模型
mlpModule.fit(
    n_output_node=2,  # 输出分几类
    input_shape=(4, 2),  # 输入的形状
    train_data=X_train,  # 训练集
    test_data=y_train,  # 验证集
    time_limit=24 * 60 * 60)  # 超时时间的设置
