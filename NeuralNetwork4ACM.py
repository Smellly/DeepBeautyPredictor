# -*- coding: utf-8 -*-
'''
// Author: Jay Smelly.
// Last modify: 2017-05-08 19:49:53.
// File name: NeuralNetwork4ACM.py
//
// Description:
    Deep Face Beautification paper reproduction
    using Tensorflow
'''
import tensorflow as tf
import numpy as np

# 添加层
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
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

def load_data(genre, batch_size=100):
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
            xs.append(x)
            ys.append(y)
        except:
            pass
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    num_examples = len(xs)
    assert(num_examples > 0)
    print('num_examples:',num_examples)
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [xs, ys], batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch, num_examples

# 1.训练的数据
lr = 0.001
dropout = 0.3
Weights_decay = 0.1
max_iteration = 20000
batch_size = 100
display_step = 100

# Make up some real data 
xs_batch, ys_batch, num_examples = load_data(genre = 'train', batch_size = batch_size)

# 2.定义节点准备接收数据
# define placeholder for inputs to network  
xs = tf.placeholder(tf.float32, [None, 136])
ys = tf.placeholder(tf.float32, [None, 1])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元   
l1 = add_layer(xs, 136, 800, 'l1', activation_function=tf.nn.relu, dropout=dropout)
l2 = add_layer(l1, 800, 800, 'l2', activation_function=tf.nn.relu, dropout=dropout)
l3 = add_layer(l2, 800, 300, 'l3', activation_function=tf.nn.relu, dropout=dropout)

# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l3, 300, 2, 'output', activation_function=None)

# 4.定义 loss 表达式
# the error between prediciton and real data    
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                     reduction_indices=[1]))

# 5.选择 optimizer 使 loss 达到最小                   
# 这一行定义了用什么方式去减少loss，学习率是 0.001    
train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# important step 对所有变量进行初始化
init = tf.initialize_all_variables()

# 迭代 20000 次学习，sess.run optimizer
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    '''
    print
    print(ys_batch)
    print
    '''
    for epoch in range(max_iteration):
        avg_cost = 0.
        total_batch = int(num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: xs_batch,
                                                          y: ys_batch})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    # '''

    # 用 saver 将所有的 variable 保存到定义的路径
    saver = tf.train.Saver()

    # 用 saver 将所有的 variable 保存到定义的路径
    save_path = saver.save(sess, "../models/save_net.ckpt")
    print("Save to path: ", save_path)

    print('Computing accuracy in val set')
    v_xs_batch, v_ys_batch ,_ = load_data(genre = 'val', batch_size = batch_size)
    print(compute_accuracy(v_xs_batch, v_ys_batch))


