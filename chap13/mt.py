import torch as t


class MTLnet(t.nn.Module):  # 多任务网络
    def __init__(self, f_size, s_size,
                 o_size):  # f_size是每个子任务的神经元，s_size是共享的神经元
        super(MTLnet, self).__init__()
        self.feature_size = f_size  # 子任务元素数
        self.shared_size = s_size  # 共享层元素数
        self.output_size = o_size  # 输出层元素数
        self.sharedlayer = t.nn.Sequential(  # 共享层
            t.nn.Linear(self.feature_size, self.shared_size), t.nn.ReLU(),
            t.nn.Dropout())
        self.task1 = t.nn.Sequential(  # 任务1，全连接网络
            t.nn.Linear(self.shared_size, self.output_size), t.nn.ReLU())
        self.task2 = t.nn.Sequential(  # 任务2，卷积网络
            t.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            t.nn.BatchNorm2d(32), t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=2, stride=1))

    def forward(self, x):
        layer_shared = self.sharedlayer(x)  # 共享部分
        out1 = self.task1(layer_shared)  # 任务1独立部分
        out2 = self.task2(layer_shared)  # 任务2独立部分
        return out1, out2


model = MTLnet(1000, 100, 10)  # 子任务1000个神经元，共享100个，输出10个
print(model)
