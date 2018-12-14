import torch as t


class AlexNet(t.nn.Module): # Alex网络
    def __init__(self): # 初始化传入输入
        super(AlexNet, self).__init__()
        # 层1
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2),
            t.nn.LocalResponseNorm(1),
            t.nn.ReLU()) # 输入深度3，输出深度96。从3,224,224压缩为96,55,55
        # 层2
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            t.nn.LocalResponseNorm(1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=3, stride=2))         # 输入深度96，输出深度256。96,55,55压缩到128,27,27
        # 层3
        self.conv3 = t.nn.Sequential(
            t.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            t.nn.LocalResponseNorm(1),
            t.nn.ReLU()) # 输入深度256，输出深度384。128,27,27压缩到384,13,13
        # 层4
        self.conv4 = t.nn.Sequential(
            t.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            t.nn.LocalResponseNorm(1),
            t.nn.ReLU()) # 输入深度384，输出深度384。384,13,13压缩到384,13,13
        # 层5
        self.conv5 = t.nn.Sequential(
            t.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            t.nn.LocalResponseNorm(1),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=3, stride=2)) # 输入深度384，输出深度256。384,13,13压缩到256,6,6

        self.dense1 = t.nn.Sequential(
            t.nn.Linear(6*6*256, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(p=0.5)
        ) # 第1个全连接层，输入6*6*256，输出4096，Dropout 0.5
        self.dense2 = t.nn.Sequential(
            t.nn.Linear(4096, 4096),
            t.nn.ReLU(),
            t.nn.Dropout(p=0.5)
        ) # 第2个全连接层，输入4096，输出4096，Dropout 0.5
        self.dense3 = t.nn.Linear(4096,1000) # 第3个全连接层，输入128，输出10类

    def forward(self, x): # 传入计算值的函数，真正的计算在这里
        x = self.conv1(x) # 3,224,224
        x = self.conv2(x) # 96,55,55
        x = self.conv3(x) # 256,27,27
        x = self.conv4(x) # 384,13,13
        x = self.conv5(x) # 384,13,13

        x = x.view(x.size(0),-1) # 

        x = self.dense1(x) # 256*6*6 -> 4096
        x = self.dense2(x) # 4096 -> 4096
        x = self.dense3(x) # 4096 -> 1000
        return x

model = AlexNet() # 建模
print(model) # 打印模型
