const tf = require('@tensorflow/tfjs');

// 加载绑定
require('@tensorflow/tfjs-node');

const model = tf.sequential();
model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    filters: 32,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
})); // 第1个卷积层，32个卷积核，大小为3*3，输入形状为(28,28,1)
model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2,
    padding: 'same',
})); // 第1个池化层
model.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
})); // 第2个卷积层，64个卷积核
model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2,
    padding: 'same',
})); // 第2个池化层
model.add(tf.layers.conv2d({
    filters: 128,
    kernelSize: 3,
    activation: 'relu',
    padding: 'same',
})); // 第3个卷积层，128个卷积核
model.add(tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2,
    padding: 'same',
})); // 第3个池化层

model.add(tf.layers.flatten()); // tfjs中的flatten()与Keras中的Flatten()同义
model.add(tf.layers.dense({ units: 625, activation: 'relu' })); // 第1个全连接层
model.add(tf.layers.dropout({ rate: 0.5 })); // 第1个dropout层
model.add(tf.layers.dense({ units: 10, activation: 'softmax' })); // 输出层

model.compile({
    optimizer: 'rmsprop', //优化方法选rmsprop
    loss: 'categoricalCrossentropy', //交叉熵
    metrics: ['accuracy'], //显示准确率
});

model.summary(); //输出模型的汇总信息

// 先生成200张28*28的随机图片
const rand1 = tf.randomUniform([200, 28, 28], 0, 1, 'float32');
rand1.print();
// 整型成卷积函数需要的格式：
const images = tf.reshape(rand1, [-1, 28, 28, 1]);

// 再生成200个label:
const labels = tf.multinomial([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 200);
labels.print();

const labels2 = tf.oneHot(labels, 10);
labels2.print();


model.fit(images, labels2, {
    epochs: 200,
    callbacks: {
        onEpochEnd: async(epoch, log) => {
            console.log(`Epoch ${epoch}: loss = ${log.loss}`);
        }
    }
});