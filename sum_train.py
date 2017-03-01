import tensorflow as tf
from contrib.utils.file import read_csv
import numpy as np



data_path = "/home/thebzeera/PycharmProjects/rztdl/contrib/utils/sum.csv"

train_data, train_label,validate_data,validate_label, test_data,test_label = read_csv(data_path, split_ratio=[60,20,20],
                                                          delimiter=',',label_vector=False
                                                                    )
# print(train_data)
# print(train_label)

learning_rate = 0.01
epochs = 500
display_step=10



input_data = tf.placeholder(tf.float32, [None, 2], name='input')

output_data = tf.placeholder(tf.float32, [None, 1], name='Output')


weights = {
    'weight1': tf.Variable(tf.random_normal([2, 32])),
    'weight2': tf.Variable(tf.random_normal([32, 64])),
    'weight3': tf.Variable(tf.random_normal([64, 1])),

}

bias = {
    'bias1': tf.Variable(tf.random_normal([32])),
    'bias2': tf.Variable(tf.random_normal([64])),
    'bias3': tf.Variable(tf.random_normal([1])),

}


def   model(x1, weights, bias):
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x1, weights['weight1']), bias['bias1']))
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['weight2']), bias['bias2']))
        layer_out = tf.add(tf.matmul(layer2, weights['weight3']), bias['bias3'])

        return layer_out


pred = model(input_data, weights, bias)

cost = tf.reduce_mean(tf.square(output_data - pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
def accuracy(pred, actual, threshold=0):
    acc = 0
    for i, j in list(zip(pred, actual)):
        i, j = int(i[0]), int(j[0])
        if i in range(j - threshold, j + threshold):
            acc += 1

    return (acc / len(pred)) * 100

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for each_epoch in range(epochs):
        loss, data, _ = sess.run([cost, pred, optimizer],
                                 feed_dict={ input_data: train_data,
                                           output_data: train_label})


        if each_epoch % display_step == 0:
            acc = accuracy(data.tolist(), train_label)
            print("Iter =", each_epoch, ", Loss= ", loss, ", Training Accuracy= ", acc)

