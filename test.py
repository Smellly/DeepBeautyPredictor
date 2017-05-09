# -*- coding: utf-8 -*-
'''
// AUTHOr: Jay Smelly.
// Last modify: 2017-05-08 19:49:53.
// File name: NeuralNetwork4ACM.py
//
// Description:
    Deep fACE bEAUTIFICATION PAPER Reproduction
    using Tensorflow
'''
import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None, dropout=1):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    # here to dropout
    # 在 Wx_plus_b 上drop掉一定比例
    # keep_prob 保持多少不被drop，在迭代时在 sess.run 中 feed
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=0.3)

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.histogram_summary(layer_name + '/outputs', outputs)  
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(correct_prediction.eval())
    print(correct_prediction.get_shape())
    count = 0
    for i,j in zip(v_ys, accuracy):
        print(i, j)
        count += 1
        if count == 10:
            break
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

def load_data(genre):
    if genre == 'train':
        with open('../data/train.txt', 'r') as fin:
            datalist = fin.readlines()
    elif genre == 'val':
        with open('../data/val.txt', 'r') as fin:
            datalist = fin.readlines()
    else:
        print('genre is wrong\nEither \'train\' or \'val\'')
        return

    filenames = [i.split()[0] for i in datalist]
    # filename_queue = tf.train.string_input_producer(filenames, shuffle=True)
    labels = [1 if int(i.split()[1].strip())>3 else 0 for i in datalist]
    xs = []
    ys = []
    # filter broken data in datalist
    for filename, y in zip(filenames, labels):
        try:
            # print('../data/face_landmark_all/'+filename.replace('jpg','npy'))
            x = np.load('../data/face_landmark_all/'+filename.replace('jpg','npy'))
            xs.append(np.reshape(x,x.shape[0]*x.shape[1]))
            ys.append(y)
        except:
            pass
    num_examples = len(xs)
    assert(num_examples > 0)
    print('num_examples:',num_examples)
    xs_mat = np.zeros((num_examples, len(xs[0])))
    ys_mat = np.zeros((num_examples, 1))
    for line in range(num_examples):
        xs_mat[line] = xs[line]	
        ys_mat[line] = ys[line]
    return xs_mat, ys_mat, num_examples

dropout = 1
xs = tf.placeholder(tf.float32, [None, 136])
ys = tf.placeholder(tf.float32, [None, 1])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 800 个神经元   
l1 = add_layer(xs, 136, 800, 'l1', activation_function=tf.nn.relu, dropout=dropout)
l2 = add_layer(l1, 800, 800, 'l2', activation_function=tf.nn.relu, dropout=dropout)
l3 = add_layer(l2, 800, 300, 'l3', activation_function=tf.nn.relu, dropout=dropout)
# add output layer 输入值是隐藏层 l3，在预测层输出 3 个结果
prediction = add_layer(l3, 300, 2, 'output', activation_function=None)

init = tf.global_variables_initializer()

def data_iterator(xs, ys, batch_size):
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(xs))
        np.random.shuffle(idxs)
        shuf_features = xs[idxs]
        shuf_labels = ys[idxs]
        for batch_idx in range(0, len(xs), batch_size):
            xs_batch = shuf_features[batch_idx:batch_idx+batch_size] / float(batch_size-1)
            xs_batch = xs_batch.astype("float32")
            ys_batch = shuf_labels[batch_idx:batch_idx+batch_size]
            yield xs_batch, ys_batch

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.import_meta_graph('../models/deep_beauty_predictor_net.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint('../models'))
    print('Computing accuracy in val set')
    v_xs_batch, v_ys_batch, _ = load_data(genre = 'val')
    print(compute_accuracy(v_xs_batch, v_ys_batch))


