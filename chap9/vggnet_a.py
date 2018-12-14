class VggNetA(t.nn.Module):  # VggNet网络A
    def __init__(self):  # 初始化传入输入
        super(VggNetA, self).__init__()
        # 层1 (1 conv + 1 pool)
        self.conv1 = t.nn.Sequential(
            t.nn.Conv2d(3, 64, kernel_size=3), t.nn.ReLU())  # 输入深度3，输出深度64
        self.pool1 = t.nn.MaxPool2d(kernel_size=2, stride=2)

        # 层2 (1 conv + 1 pool)
        self.conv2 = t.nn.Sequential(
            t.nn.Conv2d(64, 128, kernel_size=3), t.nn.ReLU())  # 输入深度64，输出深度128

        self.pool2 = t.nn.MaxPool2d(kernel_size=2, stride=2)

        # 层3_1 (2 conv + 1 pool)
        self.conv3_1 = t.nn.Sequential(
            t.nn.Conv2d(128, 256, kernel_size=3),
            t.nn.ReLU())  # 输入深度128，输出深度256

        # 层3_2
        self.conv3_2 = t.nn.Sequential(
            t.nn.Conv2d(256, 256, kernel_size=3),
            t.nn.ReLU())  # 输入深度256，输出深度256

        self.pool3 = t.nn.MaxPool2d(kernel_size=2, stride=2)

        # 层4_1 (2 conv + 1 pool)
        self.conv4_1 = t.nn.Sequential(
            t.nn.Conv2d(256, 512, kernel_size=3),
            t.nn.ReLU())  # 输入深度256，输出深度512

        # 层4_2
        self.conv4_2 = t.nn.Sequential(
            t.nn.Conv2d(512, 512, kernel_size=3),
            t.nn.ReLU())  # 输入深度512，输出深度512

        self.pool4 = t.nn.MaxPool2d(kernel_size=2, stride=2)

        # 层5_1 (2 conv + 1 pool)
        self.conv5_1 = t.nn.Sequential(
            t.nn.Conv2d(512, 512, kernel_size=3),
            t.nn.ReLU())  # 输入深度512，输出深度512

        # 层5_2
        self.conv5_2 = t.nn.Sequential(
            t.nn.Conv2d(512, 512, kernel_size=3),
            t.nn.ReLU())  # 输入深度512，输出深度512

        self.pool5 = t.nn.MaxPool2d(kernel_size=2, stride=2)

        # 第1个全连接层，输入6*6*256，输出4096，Dropout 0.5
        self.dense1 = t.nn.Sequential(
            t.nn.Linear(6 * 6 * 256, 4096), t.nn.ReLU(), t.nn.Dropout(p=0.5))

        # 第2个全连接层，输入4096，输出4096，Dropout 0.5
        self.dense2 = t.nn.Sequential(
            t.nn.Linear(4096, 4096), t.nn.ReLU(), t.nn.Dropout(p=0.5))

        self.dense3 = t.nn.Linear(4096, 1000)  # 第3个全连接层，输入128，输出10类

    # 传入计算值的函数，真正的计算在这里
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.pool5(x)

        x = x.view(x.size(0), -1)  #

        x = self.dense1(x)  #  -> 4096
        x = self.dense2(x)  # 4096 -> 4096
        x = self.dense3(x)  # 4096 -> 1000
        return x
