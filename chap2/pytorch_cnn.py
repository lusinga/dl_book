import numpy as np
import torch as t


# 数据读取部分
def read_labels(filename, items):  # 读取图片对应的数字
    file_labels = open(filename, 'rb')
    file_labels.seek(8)
    data = file_labels.read(items)
    y = np.zeros(items, dtype=np.int64)
    for i in range(items):
        y[i] = data[i]
    file_labels.close()
    return y


y_train = read_labels('./train-labels-idx1-ubyte', 60000)
y_test = read_labels('./t10k-labels-idx1-ubyte', 10000)


def read_images(filename, items):  # 读取图像
    file_image = open(filename, 'rb')
    file_image.seek(16)

    data = file_image.read(items * 28 * 28)

    X = np.zeros(items * 28 * 28, dtype=np.float32)
    for i in range(items * 28 * 28):
        X[i] = data[i] / 255
    file_image.close()
    return X.reshape(-1, 28 * 28)


X_train = read_images('./train-images-idx3-ubyte', 60000)
X_test = read_images('./t10k-images-idx3-ubyte', 10000)

# 超参数
num_epochs = 30  # 训练轮数
learning_rate = 1e-3  # 学习率
batch_size = 100  # 批量大小


class FirstCnnNet(t.nn.Module):  # CNN网络
    def __init__(self, num_classes):  # 初始化只需要输出这一个参数
        super(FirstCnnNet, self).__init__()
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            t.nn.BatchNorm2d(16),
            t.nn.ReLU())  # 输入深度1，输出深度16。从1,28,28压缩为16,14,14
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            t.nn.BatchNorm2d(32), t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=2,
                           stride=1))  # 输入深度16，输出深度32。16,14,14压缩到32,6,6
        self.dense1 = t.nn.Sequential(
            t.nn.Linear(6 * 6 * 32, 128), t.nn.ReLU(),
            t.nn.Dropout(p=0.25))  # 第1个全连接层，输入6*6*32，输出128
        self.dense2 = t.nn.Linear(128, num_classes)  # 第2个全连接层，输入128，输出10类

    def forward(self, x):  # 传入计算值的函数，真正的计算在这里
        x = self.conv1(x)  # 16,14,14
        x = self.conv2(x)  # 32,6,6
        x = x.view(x.size(0), -1)
        x = self.dense1(x)  # 32*6*6 -> 128
        x = self.dense2(x)  # 128 -> 10
        return x


model = FirstCnnNet(10)  # 输入28*28，隐藏层128，输出10类
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)  # 优化器仍然选随机梯度下降

X_train_size = len(X_train)

for epoch in range(num_epochs):
    print('Epoch:', epoch)  # 打印轮次：

    X = t.autograd.Variable(t.from_numpy(X_train))
    y = t.autograd.Variable(t.from_numpy(y_train))

    i = 0
    while i < X_train_size:
        X0 = X[i:i + batch_size]  #取一个新批次的数据
        X0 = X0.view(-1, 1, 28, 28)
        y0 = y[i:i + batch_size]
        i += batch_size

        # 正向传播
        out = model(X0)  # 用神经网络计算10类输出结果
        loss = t.nn.CrossEntropyLoss()(out, y0)  # 计算神经网络结果与实际标签结果的差值

        # 反向梯度下降
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 根据误差函数求导
        optimizer.step()  # 进行一轮梯度下降计算

    print(loss.item())

# 验证部分

model.eval()  ## 将模型设为验证模式

X_val = t.autograd.Variable(t.from_numpy(X_test))
y_val = t.autograd.Variable(t.from_numpy(y_test))

X_val = X_val.view(-1, 1, 28, 28)  # 整形成CNN需要的输入

out_val = model(X_val)  ## 用训练好的模型计算结果
loss_val = t.nn.CrossEntropyLoss()(out_val, y_val)

print(loss_val.item())

_, pred = t.max(out_val, 1)  # 求出最大的元素的位置
num_correct = (pred == y_val).sum()  # 将预测值与标注值进行对比

print(num_correct.data.numpy() / len(y_test))
