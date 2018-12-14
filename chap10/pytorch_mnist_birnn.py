import numpy as np
import torch as t


# 读取图片对应的数字
def read_labels(filename, items):
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


# 读取图像
def read_images(filename, items):
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
num_epochs = 30 # 训练轮数
learning_rate = 1e-3 # 学习率
batch_size = 100 # 批量大小


# 双向LSTM 网络
class BiLstmNet(t.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLstmNet, self).__init__()
        self.hidden_size = hidden_size # 隐藏单元数
        self.num_layers = num_layers # 隐藏层数
        self.lstm = t.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True) # 双向LSTM
        self.fc = t.nn.Linear(hidden_size*2, num_classes) # 输出的全连接网络

    def forward(self, x):
        # 通过x.size(0)获取batch中的元素个数
        b_size = x.size(0)# h0和c0的格式为：(层数 * 方向数, 批次数, 隐藏层数)

        h0 = t.zeros(self.num_layers*2, b_size, self.hidden_size)
        c0 = t.zeros(self.num_layers*2, b_size, self.hidden_size)

        lstm_out, _ = self.lstm(x, (h0, c0)) # 双向lstm输出
        fc_out = self.fc(lstm_out[:, -1, :]) # 分类输出
        return fc_out

model = BiLstmNet(28 * 28, 128, 1, 10) # 输入28*28，隐藏元素128，隐藏层1个，输出10类
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate) # 优化器仍然选随机梯度下降

X_train_size = len(X_train)

for epoch in range(num_epochs):
    print('Epoch:', epoch) # 打印轮次

    X = t.autograd.Variable(t.from_numpy(X_train))
    y = t.autograd.Variable(t.from_numpy(y_train))

    i = 0
    while i < X_train_size:
        X0 = X[i:i + batch_size] # 取一个新批次的数据
        X0 = X0.view(batch_size, -1, 28 * 28)         # LSTM要求的格式为：batch_size,seq, input_size

        y0 = y[i:i + batch_size]
        i += batch_size

        # 正向传播
        out = model(X0) # 用神经网络计算10类输出结果
        loss = t.nn.CrossEntropyLoss()(out, y0) # 计算神经网络结果与实际标签结果的差值

        # 反向梯度下降
        optimizer.zero_grad() # 清空梯度
        loss.backward() # 根据误差函数求导
        optimizer.step() # 进行一轮梯度下降计算

    print(loss.item())

# 验证部分
model.eval() # 将模型设为验证模式

X_val = t.autograd.Variable(t.from_numpy(X_test))
y_val = t.autograd.Variable(t.from_numpy(y_test))
X_val = X_val.view(10000, -1, 28 * 28) # 整形成CNN需要的结果
out_val = model(X_val) # 用训练好的模型计算结果
loss_val = t.nn.CrossEntropyLoss()(out_val, y_val)

print(loss_val.item())
_, pred = t.max(out_val, 1) # 求出最大的元素的位置
num_correct = (pred == y_val).sum() # 将预测值与标注值进行对比

print(num_correct.data.numpy() / len(y_test))
