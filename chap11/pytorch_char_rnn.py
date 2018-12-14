import torch as t
import string
import random

# 输入文本文件
filename = './input.txt'

# 超参数
num_epochs = 20000 # 轮数
hidden_size = 100 # 隐藏层大小
num_layers = 2 # 隐藏层层数
learning_rate = 1e-2 # 学习率
sentence_length = 200 # 句子长度
batch_size = 100 # 批量大小

all_printable = string.printable # 所有可打印的字符
num_printable = len(all_printable) # 所有可打印字符数

progress_every = 100 # 输出间隔


def tensor_to_char(tensor): # 向量转成字符
    value = tensor.item()
    return all_printable[value]


def char_to_tensor(chars): # 将字符串转化成向量

    c_length = len(chars)

    c_tensor = t.zeros(c_length, dtype=t.int64)
    for ch in range(c_length):
        try:
            c_tensor[ch] = all_printable.index(chars[ch])
        except ValueError:
            continue
    return c_tensor


def get_training_set(c_len, bch_size): # 从文本中随机选取一段进行训练
    x = t.zeros(bch_size, c_len, dtype=t.int64) # 输入向量
    y = t.zeros(bch_size, c_len, dtype=t.int64) # 标签向量

    for bi in range(bch_size):
        start_index = random.randint(0, file_len - c_len - 1) # 随机生成一个起始位置
        end_index = start_index + c_len + 1 # 长度为c_len + 1。因为y[start_index + c_len]的值实际上是start_index+ c_len+1
        chunk = input_file[start_index:end_index] # chunk取得的长度是c_len + 1，比x[bi]和y[bi]都多1
        x_i = chunk[:-1] # x不需要c_len位置的字符
        y_i = chunk[1:] # y的第一个是x的第2个，所以不需要0位置的字符

        x[bi] = char_to_tensor(x_i) # 转换成编码后的向量
        y[bi] = char_to_tensor(y_i) # 转换成编码后的向量
    
    x = t.autograd.Variable(x) # 返回变量
    y = t.autograd.Variable(y)
    return x, y


class LstmNet(t.nn.Module): # Lstm 网络
    def __init__(self, input_size, h_size, n_layers, num_classes):
        super(LstmNet, self).__init__()
        self.hidden_size = h_size # 隐藏单元数
        self.num_layers = n_layers # 隐藏层数
        self.encoder = t.nn.Embedding(input_size, h_size) # 嵌入层
        self.lstm = t.nn.LSTM(h_size, h_size, n_layers, batch_first=True) # LSTM层
        self.fc = t.nn.Linear(h_size, num_classes) # 输出的全连接网络

    def forward(self, x, hidden_state):
        # 调用模型时需要指定隐藏状态
        encoded = self.encoder(x) # 首先将输入映射成词向量

        bch_size = x.size(0)
        lstm_out, (h2, c2) = self.lstm(
            encoded.view(bch_size, 1, -1), hidden_state) # 计算lstm输出
        
        c2 = c2.detach() # 通过detach清理旧状态
        h2 = h2.detach()
        hidden_state2 = (h2, c2) # 保存隐藏状态

        fc_out = self.fc(lstm_out.view(bch_size, -1))
        return fc_out, hidden_state2

    def init_hidden(self, bch_size): # 初始化h0 , c0. 格式为：(层数 * 方向数, 批次数, 隐藏层数)
        
        return t.zeros(self.num_layers, bch_size, self.hidden_size), t.zeros(
            self.num_layers, bch_size, self.hidden_size)


def generate(decoder,
             all_characters,
             seed_string,
             generate_len=100,
             temperature=100): # 生成序列的函数
    hidden = decoder.init_hidden(1) # 初始化1字符大小的LSTM网络
    input_seed = t.autograd.Variable(char_to_tensor(seed_string).unsqueeze(0))

    predicted = seed_string # 返回值的头部是输入的种子

    for p in range(len(seed_string) - 1): # 使用输入的种子字符串来构建隐藏状态
        _, hidden = decoder(input_seed[:, p], hidden)

    inp = input_seed[:, -1] # 输入向量

    for p in range(generate_len):
        out, hidden = decoder(inp, hidden)

        output_dist = out.data.view(-1).div(temperature).exp() # 进行温度值计算
        top_i = t.multinomial(output_dist, 1).item() # 取多项分布采样的第1个值

        predicted_char = all_characters[top_i] # 拼接完整预测字符串
        predicted += predicted_char
        inp = t.autograd.Variable(char_to_tensor(predicted_char).unsqueeze(0)) # 更新预测输入值

    return predicted


input_file = open(filename).read() # 读取文件
file_len = len(input_file)

# 模型、优化器和损失函数

model = LstmNet(num_printable, hidden_size, num_layers, num_printable) # LSTM模型
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate) # 优化方法选Adam
criterion = t.nn.CrossEntropyLoss() # 交叉熵
for epoch in range(1, num_epochs + 1):
    hidden_train = model.init_hidden(batch_size) # 初始化隐藏值
    model.zero_grad() # 清0梯度
    loss = 0
    x_tensor, y_tensor = get_training_set(sentence_length, batch_size) # 获取随机文本进行训练

    for c in range(sentence_length):
        output, hidden_train = model(x_tensor[:, c], hidden_train) # 计算模型
        loss += criterion(output.view(batch_size, -1), y_tensor[:, c])

    loss.backward() # 梯度求导
    optimizer.step() # 梯度计算

    loss = loss.item() / sentence_length

    if epoch % progress_every == 0:
        print('轮次:', epoch, " 总轮次 ", num_epochs)
        print('损失值:', loss)
        print('生成的句子为:')
        print(
            generate(model, all_printable, seed_string='The', generate_len=100))

t.save(model, "input.pt") # 保存模型参数
print(generate(model, all_printable, seed_string='What', generate_len=100))
