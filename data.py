# -*- coding: utf-8 -*-
'''
// Author: Jay Smelly.
// Last modify: 2017-05-08 19:49:53.
// File name: NeuralNetwork4ACM.py
//
// Description:
    Deep Face Beautification paper reproduction
    using Tensorflow
    loading data
'''

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
