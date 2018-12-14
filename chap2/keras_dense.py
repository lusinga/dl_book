import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

# 读取数据部分


# 读取图片标记，也就是要学习的数字
def read_labels(filename, items):
    file_labels = open(filename, 'rb')
    file_labels.seek(8)  # 标签文件的头是8个字节，略过不读
    data = file_labels.read(items)
    y = np.zeros(items)
    for i in range(items):
        y[i] = data[i]
    file_labels.close()
    return y


y_train = read_labels('./train-labels-idx1-ubyte', 60000)  # 读取60000张训练标记
y_test = read_labels('./t10k-labels-idx1-ubyte', 10000)  # 读取10000张测试标记


# 读取图片
def read_images(filename, items):
    file_image = open(filename, 'rb')
    file_image.seek(16)  # 图像文件的头是16个字节，略过不读

    data = file_image.read(items * 28 * 28)

    X = np.zeros(items * 28 * 28)
    for i in range(items * 28 * 28):  # 将值转换成灰度
        X[i] = data[i] / 255
    file_image.close()
    return X.reshape(-1, 28 * 28)


X_train = read_images('train-images-idx3-ubyte', 60000)  # 读取60000张训练图片
X_test = read_images('./t10k-images-idx3-ubyte', 10000)  # 读取10000张测试图片

y_train = keras.utils.to_categorical(y_train, 10)  # one hot转码
y_test = keras.utils.to_categorical(y_test, 10)

# 训练与验证部分

model = Sequential()  # 构建模型

model.add(Dense(units=121, input_dim=28 * 28))  # 输入28*28, 输出121
model.add(Activation('relu'))  # 激活函数为relu
model.add(Dense(units=10))  # 分为10类
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) # 编译模型

model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=20,
    verbose=1,
    validation_data=(X_test, y_test))  # 进行训练
score = model.evaluate(X_test, y_test, verbose=1)  # 验证

model.save('keras_mnist1.model') # 保存模型参数

print('损失值:', score[0])
print('准确率:', score[1])
