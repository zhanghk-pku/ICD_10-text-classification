import tensorflow as tf
import numpy as np

class Models(object):
    def __init__(self, vocab_size, num_classes, word_emb_dim, feature_emb_dim, add_feature, seq_length,
                 hidden_dim, num_features, keep_prob, add_keyword_attention):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.word_emb_dim = word_emb_dim
        self.feature_emb_dim = feature_emb_dim
        self.add_feature = add_feature
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        self.keep_prob = keep_prob
        self.add_keyword_attention = add_keyword_attention

    def lstm_model(self, train_x, train_y, test_x, test_y, keyword_id, mask_train, mask_test,
                   num_layers, batch_size, learning_rate,num_epochs):
        tf.reset_default_graph()
        word_embeddings = tf.get_variable('word_embeddings', shape=[self.vocab_size + 1, self.word_emb_dim],
                                          initializer=tf.truncated_normal_initializer(stddev=0.01),dtype=tf.float32)
        mask_x = tf.placeholder(tf.float32, [None, self.seq_length, self.hidden_dim])
        input_y = tf.placeholder(tf.float32, [None, self.num_classes])

        with tf.variable_scope('embedding_layer'):
            if self.add_feature:
                input_x = tf.placeholder(tf.int32, [None, self.seq_length, 2])
                feature_embeddings = tf.get_variable('feature_embeddings',[self.num_features + 1, self.feature_emb_dim],
                                                     initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                     dtype=tf.float32)
                input_word_emb = tf.nn.embedding_lookup(word_embeddings, input_x[:, :, 0])
                input_feature_emb = tf.nn.embedding_lookup(feature_embeddings, input_x[:, :, 1])
                input_emb = tf.concat([input_word_emb, input_feature_emb], 2)
            else:
                input_x = tf.placeholder(tf.int32, [None, self.seq_length])
                input_emb = tf.nn.embedding_lookup(word_embeddings, input_x)

        with tf.variable_scope('first_lstm'):
            cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
            dropout_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
            multi_dropout_cell = [dropout_cell for _ in range(num_layers)]
            multi_lstm_cell = tf.contrib.rnn.MultiRNNCell(multi_dropout_cell, state_is_tuple=True)
            outputs_1, _ = tf.nn.dynamic_rnn(cell=multi_lstm_cell, inputs=input_emb, dtype=tf.float32)
            first_outputs = outputs_1 * mask_x

        if self.add_keyword_attention:
            keyword_emb = tf.nn.embedding_lookup(word_embeddings, keyword_id)
            with tf.variable_scope('attention'):
                atten_weight = tf.get_variable('atten_weight',[self.word_emb_dim, self.hidden_dim],
                                               initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               dtype=tf.float32)
                trans_keyword_emb = tf.matmul(keyword_emb, atten_weight)
                atten_dot = tf.tensordot(first_outputs, tf.transpose(trans_keyword_emb), axes=1)
                alphas = tf.nn.softmax(atten_dot)
                atten_outputs = tf.reduce_sum(tf.expand_dims(alphas, -1) * trans_keyword_emb, 2)
                print(atten_outputs.shape)

            with tf.variable_scope('concate'):
                concate_inputs = tf.concat([input_emb, atten_outputs], 2)
                print(concate_inputs.shape)
                print(concate_inputs.get_shape())

            with tf.variable_scope('second_lstm'):
                cell_2 = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
                dropout_cell_2 = tf.contrib.rnn.DropoutWrapper(cell_2, output_keep_prob=self.keep_prob)
                multi_dropout_cell_2 = [dropout_cell_2 for _ in range(num_layers)]
                multi_lstm_cell_2 = tf.contrib.rnn.MultiRNNCell(multi_dropout_cell_2, state_is_tuple=True)
                second_outputs, _ = tf.nn.dynamic_rnn(cell=multi_lstm_cell_2, inputs=concate_inputs, dtype=tf.float32)
                last = tf.reduce_mean(second_outputs * mask_x, axis=1)
        else:
            last = tf.reduce_mean(first_outputs, axis=1)

        with tf.variable_scope('full_connect_layer'):
            # fc_1 = tf.layers.dense(last, self.hidden_dim,activation=tf.nn.relu)
            logit = tf.layers.dense(last, self.num_classes)
            y_pred = tf.argmax(tf.nn.softmax(logit), 1)

        with tf.variable_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=input_y)
            loss = tf.reduce_mean(cross_entropy)
            optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.variable_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(input_y, 1), y_pred)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for t in range(num_epochs):
                for i in range(int(len(train_x) / batch_size)+1):
                    k = batch_size * i
                    h = k + batch_size

                    train_pred_y, train_loss, _ = sess.run([y_pred, loss, optim], feed_dict={input_x: train_x[k:h], input_y: train_y[k:h],
                                               mask_x: mask_train[k:h]})
                    if i % 100 == 0:
                        print('batch {}, training loss = {}'.format(i, train_loss))
                        print('   train:')
                        print('   true label:', np.argmax(train_y[k:k + 15], axis=1))
                        print('   pred label:', train_pred_y[:15])
                    if i % 100 == 0:
                        test_acc = []
                        test_loss = []
                        for j in range(int(len(test_x) / batch_size)+1):
                            m = batch_size * j
                            n = m + batch_size
                            t_loss, accuracy, test_pred_y = sess.run([loss, acc, y_pred], feed_dict={input_x: test_x[m:n], input_y: test_y[m:n],
                                                                mask_x: mask_test[m:n]})
                            if j % 100 == 0:
                                print('   test:')
                                print('   true label:', np.argmax(test_y[m:m+15], axis=1))
                                print('   pred label:', test_pred_y[:15])
                            test_acc.append(accuracy)
                            test_loss.append(t_loss)
                        print('   test loss = {}, test accuracy is {}'.format(np.mean(test_loss), np.mean(test_acc)))
                        print('-----------------------------------------------------')
