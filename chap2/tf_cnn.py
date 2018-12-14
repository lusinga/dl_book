import tensorflow as tf
import numpy as np

# 读取数据代码请参见Keras和PyTorch部分，节约篇幅这里就不重复了

batch_size = 128  # 训练批次
test_size = 256  # 测试批次


def model(X, filter1, filter2, filter3, fc_weight, out_weight):  # 建模
    # 第一个卷积核: [3,3,1,32]
    conv1_1 = tf.nn.conv2d(
        X, filter1, strides=[1, 1, 1, 1],
        padding='SAME')  # 输入的形状： (?, 28, 28, 32)
    conv1 = tf.nn.relu(conv1_1)
    pool1 = tf.nn.max_pool(
        conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')  # 池化后的形状(?, 14, 14, 32)

    # 第二个卷积核: [3,3,32,64]
    conv2_1 = tf.nn.conv2d(
        pool1, filter2, strides=[1, 1, 1, 1],
        padding='SAME')  # 输入形状 (?, 14, 14, 32)
    conv2 = tf.nn.relu(conv2_1)
    pool2 = tf.nn.max_pool(
        conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')  # 池化后的形状：(?, 7, 7, 64)

    # 第三个卷积核: [3,3,64,128]
    conv3_1 = tf.nn.conv2d(
        pool2, filter3, strides=[1, 1, 1, 1],
        padding='SAME')  # 输入形状 (?, 7, 7, 128)
    conv3 = tf.nn.relu(conv3_1)
    pool3 = tf.nn.max_pool(
        conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
        padding='SAME')  # 池化后的形状(?, 4, 4, 128)

    flatten = tf.reshape(
        pool3, [-1, fc_weight.get_shape().as_list()[0]])  # 变形成 (?, 2048)

    fc = tf.nn.relu(tf.matmul(flatten, fc_weight))  # 全连接层

    out = tf.matmul(fc, out_weight)  # 输出层
    return out


X = tf.placeholder("float", [None, 28, 28, 1])  # 占位符，描述用
Y = tf.placeholder("float", [None, 10])

# 下面是变量的定义
filter1 = tf.Variable(
    tf.random_normal([3, 3, 1, 32], stddev=0.01),
    name='filter1')  # 第一个卷积核: [3,3,1,32]
filter2 = tf.Variable(
    tf.random_normal([3, 3, 32, 64], stddev=0.01),
    name='filter2')  # 第二个卷积核: [3,3,32,64]
filter3 = tf.Variable(
    tf.random_normal([3, 3, 64, 128], stddev=0.01),
    name='filter3')  # 第三个卷积核: [3,3,64,128]
fc_weight = tf.Variable(
    tf.random_normal([128 * 4 * 4, 625], stddev=0.01),
    name='fc_weight')  # 全连接层参数
out_weigth = tf.Variable(
    tf.random_normal([625, 10], stddev=0.01), name='out_weight')  # 输出层参数

py_x = model(X, filter1, filter2, filter3, fc_weight, out_weigth)  # 建模

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))  # 损失函数
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)  # 优化器
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()  # 别忘了初始化变量

    for i in range(100):
        training_batch = zip(
            range(0, len(X_train), batch_size),
            range(batch_size,
                  len(X_train) + 1, batch_size))
        for start, end in training_batch:
            sess.run(
                train_op,
                feed_dict={
                    X: X_train[start:end],
                    Y: y_train[start:end],
                })  # 训练
        test_indices = np.arange(len(X_test))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        print(i,
              np.mean(
                  np.argmax(y_test[test_indices], axis=1) == sess.run(
                      predict_op, feed_dict={
                          X: X_test[test_indices],
                      })))  # 验证
