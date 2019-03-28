import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

all_words = []
all_sentences = []
with open('chief_text.txt','r',encoding='utf-8') as f:
    for line in f:
        words = line.split()
        all_sentences.append(words)
        for word in words:
            if word not in all_words:
                all_words.append(word)
vocab = {}
for i, word in enumerate(all_words,1):
    vocab[word] = i

x_data = [[vocab[word] for word in sentence] for sentence in all_sentences]
x_input = []
for sentence in x_data:
    if len(sentence) < 12:
        sentence = sentence + [0]*(12-len(sentence))
    else:
        sentence = sentence[:12]
    x_input.append(sentence)

label = pd.read_table('label.txt',names=['label'])
one_hot = pd.get_dummies(label['label'])
y_input = np.array(one_hot).astype(np.float64)

embedding_dim = 100  # 词向量维度
seq_length = 12  # 序列长度
num_classes = 119  # 类别数
vocab_size = 3108  # 词汇表达小
num_layers = 2  # 隐藏层层数
hidden_dim = 128  # 隐藏层神经元
dropout_keep_prob = 0.8  # dropout保留比例
learning_rate = 1e-3  # 学习率
batch_size = 100  # 每批训练大小
num_epochs = 5  # 总迭代轮次


def LSTM(x, y):
    tf.reset_default_graph()
    input_x = tf.placeholder(tf.int32, [None, seq_length])
    input_y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = dropout_keep_prob
    embedding = tf.get_variable('embedding', [vocab_size, embedding_dim])
    embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)

    def dropout():
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
        return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    with tf.name_scope('lstm'):
        cells = [dropout() for _ in range(num_layers)]
        lstm_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=embedding_inputs, dtype=tf.float32)
        last = outputs[:, -1, :]

    with tf.name_scope('score'):
        fc = tf.layers.dense(last, hidden_dim, name='fc1')
        fc = tf.contrib.layers.dropout(fc, keep_prob)
        fc = tf.nn.relu(fc)
        logit = tf.layers.dense(fc, num_classes, name='fc2')
        y_pred = tf.argmax(tf.nn.softmax(logit), 1)

    with tf.name_scope('optimize'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=input_y)
        loss = tf.reduce_mean(cross_entropy)
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(input_y, 1), y_pred)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    for i in range(num_epochs):
        for j in range(int(len(x) / batch_size)):
            k = batch_size * j
            h = k + batch_size
            sess.run(optim, feed_dict={input_x: x[k:h], input_y: y[k:h]})
            if j % 20 == 0:
                accuracy = sess.run(acc, feed_dict={input_x: x[:h], input_y: y[:h]})
                print('the {} epoch {} batch, the training accuracy: {}'.format(i, j, accuracy))

x = np.array(x_input)
y = y_input
LSTM(x,y)