import torch as t


class AutoEncoder(t.nn.Module):  # 自动编解码器例
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = t.nn.Sequential(
            t.nn.Linear(28 * 28, 128),  # 784 -> 128
            t.nn.ReLU(True),
            t.nn.Linear(128, 64),  # 128 -> 64
            t.nn.ReLU(True),
            t.nn.Linear(64, 12),  # 64 -> 12
            t.nn.ReLU(True),
            t.nn.Linear(12, 3)  # 12 -> 3
        )
        self.decoder = t.nn.Sequential(  # 与编码器的顺序刚好相反
            t.nn.Linear(3, 12), t.nn.ReLU(True), t.nn.Linear(12, 64),
            t.nn.ReLU(True), t.nn.Linear(64, 128), t.nn.ReLU(True),
            t.nn.Linear(128, 28 * 28), t.nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)  # 编码器
        x = self.decoder(x)  # 解码器
        return x


model = AutoEncoder()
print(model)
