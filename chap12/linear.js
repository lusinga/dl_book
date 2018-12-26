const tf = require('@tensorflow/tfjs');

// 加载绑定
require('@tensorflow/tfjs-node'); // 如果有GPU，请换成 '@tensorflow/tfjs-node-gpu'

const endValue = 100; //坐标结束值
const learningRate = 0.01; // 学习率

const xs = tf.range(0, endValue); // x是从0到endValue的序列
let rand2 = tf.randomNormal([endValue], 0, 1); // rand2给序列带来一些扰动
let ys = tf.mul(xs, 2);
ys = tf.add(ys, 1);
ys = tf.add(ys, rand2);

// w和b是两个要训练的变量
const w = tf.scalar(Math.random()).variable();
const b = tf.scalar(Math.random()).variable();


const f = function(x) {
    let result = tf.mul(x, w).add(b); // y = w * x + b
    w.print();
    b.print();
    result.print();
    return result;
};

const loss = (pred, label) => pred.sub(label).square().mean(); // 误差是均方差
const optimizer = tf.train.sgd(learningRate); // 优化器选用随机梯度下降

for (let i = 0; i < 100; i++) {
    optimizer.minimize(() => loss(f(xs), ys)); // 训练模型
}

// 进行预测
console.log(
    `w: ${w.dataSync()}, b: ${b.dataSync()}`);
const preds = f(xs).dataSync();
preds.forEach((pred, i) => {
    console.log(`x: ${i}, pred: ${pred}`);
});