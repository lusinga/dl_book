import numpy as np
import torch as t

# 数据读取部分
def read_labels(filename, items): # 读取图片对应的数字
    file_labels = open(filename, 'rb')
    file_labels.seek(8)
    data = file_labels.read(items)
    y = np.zeros(items, dtype=np.int64)
    for i in range(items):
        y[i] = data[i]
    file_labels.close()
    return y


y_train = read_labels('/Users/ziyingliuziying/lusing/mnist/train-labels-idx1-ubyte', 60000)
y_test = read_labels('/Users/ziyingliuziying/lusing/mnist/t10k-labels-idx1-ubyte', 10000)


def read_images(filename, items): # 读取图像
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
num_epochs = 1000 # 训练轮数
learning_rate = 1e-3 # 学习率
batch_size = 64 # 批量大小


class TestNet(t.nn.Module):
    def __init__(self, in_dim, hidden, out_dim): # 初始化传入输入、隐藏层、输出三个参数
        super(TestNet, self).__init__()
        self.layer1 = t.nn.Sequential(t.nn.Linear(in_dim, hidden), t.nn.ReLU(True)) # 全连接层
        self.layer2 = t.nn.Linear(hidden, out_dim) # 输出层

    def forward(self, x): # 传入计算值的函数，真正的计算在这里
        x = self.layer1(x)
        x = self.layer2(x)
        return x


model = TestNet(28 * 28, 121, 10) # 输入28*28，隐藏层121，输出10类
optimizer = t.optim.SGD(model.parameters(), lr=learning_rate) # 优化器仍然选随机梯度下降

X_train_size = len(X_train) # 训练集大小

for epoch in range(num_epochs):
    print('Epoch:',epoch) # 打印轮次：

    X = t.autograd.Variable(t.from_numpy(X_train)) # 训练的参数需要用变量来保存
    y = t.autograd.Variable(t.from_numpy(y_train))

    i = 0 # 循环控制变量
    while i < X_train_size:
        X0 = X[i:i+batch_size]        #取一个新批次的数据
        y0 = y[i:i+batch_size]
        i += batch_size

        # 正向传播
        out = model(X0) # 用神经网络计算10类输出结果
        loss = t.nn.CrossEntropyLoss()(out, y0) # 计算神经网络结果与实际标签结果的差值

        # 反向梯度下降
        optimizer.zero_grad() # 清空梯度
        loss.backward() # 根据误差函数求导
        optimizer.step() # 进行一轮梯度下降计算

    print(loss.item()) # 打印损失值

# 验证部分
model.eval() # 将模型设为验证模式

X_val = t.autograd.Variable(t.from_numpy(X_test)) # 验证变量
y_val = t.autograd.Variable(t.from_numpy(y_test))

out_val = model(X_val) # 用训练好的模型计算结果
loss_val = t.nn.CrossEntropyLoss()(out_val, y_val) # 计算交叉熵

print(loss_val.item()) # 打印测试损失值

_, pred = t.max(out_val,1) # 求出最大的元素的位置
num_correct = (pred == y_val).sum() # 将预测值与标注值进行对比

print(num_correct.data.numpy()/len(y_test)) # 打印正确率
