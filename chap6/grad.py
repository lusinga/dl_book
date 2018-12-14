import torch as t

# 超参数
lr = 1e-4  # 学习率
epoch = 1000  # 轮数

X_train = t.linspace(-10, 10, 100).view(-1, 1)
y_train = X_train * 2.0 + t.normal(
    t.zeros(100), std=0.5).view(-1, 1)  # 生成y=2x附近的随机数

# 数据放入变量中，X和y并不需要自动求导，所以不用设requires_grad
X = t.autograd.Variable(X_train)
y = t.autograd.Variable(y_train)

# 下面是要训练的参数
W = t.autograd.Variable(t.rand(1, 1), requires_grad=True)  # W是斜率，需要自动求导
b = t.autograd.Variable(t.zeros(1, 1), requires_grad=True)  # b是截距，也需要自动求导


def MSELoss2(y, out):  # 模型计算值out与y之间的差的平方和的平均值函数
    loss = y - out
    loss = loss * loss
    return loss.mean()


def model2(X, W, y):  # 模型计算，就是一个线性计算
    return X.mm(W) + b.expand_as(y)


def step():  # 进行一次优化更新
    W.data.sub_(lr * W.grad.data)
    b.data.sub_(lr * b.grad.data)


def zero_grad():  # 梯度清0
    W.grad.data.zero_()
    b.grad.data.zero_()


for i in range(epoch):
    # 正向传播

    out = model2(X, W, y)  # 模型计算
    loss = MSELoss2(out, y)  # 误差计算

    loss.backward()  # 反向梯度下降

    step()  # 更新W和b这两个参数

    zero_grad()  # 因为PyTorch的梯度是累加的，所以每次都要记得清0梯度值

print(W.data)
print(b.data)
