import tensorflow as tf
import numpy as np

class Models(object):
    def __init__(self, vocab_size, num_classes, num_features, word_emb_dim, feature_emb_dim, hidden_dim, num_layers,
                 learning_rate, keep_prob, add_feature_emb, add_second_attention, add_keyword_attention):
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.num_features = num_features
        self.word_emb_dim = word_emb_dim
        self.feature_emb_dim = feature_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.add_feature_emb = add_feature_emb
        self.add_second_attention = add_second_attention
        self.add_keyword_attention = add_keyword_attention

    def batch_seq_padding(self, sequences):
        seq_padding = []
        seq_len = [len(seq) for seq in sequences]
        max_seqlen = max(seq_len)
        for seq in sequences:
            if len(seq) < max_seqlen:
                if self.add_feature_emb:
                    seq = seq + [[0,0]] * (max_seqlen - len(seq))
                else:
                    seq = seq + [0] * (max_seqlen - len(seq))
            else:
                pass
            assert len(seq) == max_seqlen, 'the sequences length of input batch is not consistent!'
            seq_padding.append(seq)
        return seq_padding, seq_len

    def lstm_model(self, train_x, train_y, test_x, test_y, keyword_id, batch_size, num_epochs):
        tf.reset_default_graph()
        input_y = tf.placeholder(tf.float32, [None, self.num_classes])
        input_seqlen = tf.placeholder(tf.int32, [None])

        with tf.variable_scope('embedding_layer'):
            word_embeddings = tf.get_variable('word_embeddings', shape=[self.vocab_size + 1, self.word_emb_dim],
                                              initializer=tf.truncated_normal_initializer(stddev=0.01),
                                              dtype=tf.float32)
            feature_embeddings = tf.get_variable('feature_embeddings', [self.num_features + 1, self.feature_emb_dim],
                                                 initializer=tf.truncated_normal_initializer(stddev=0.01),
                                                 dtype=tf.float32)
            if self.add_feature_emb:
                input_x = tf.placeholder(tf.int32, [None, None, 2])
                input_word_emb = tf.nn.embedding_lookup(word_embeddings, input_x[:, :, 0])
                input_feature_emb = tf.nn.embedding_lookup(feature_embeddings, input_x[:, :, 1])
                input_emb = tf.concat([input_word_emb, input_feature_emb], 2)
            else:
                input_x = tf.placeholder(tf.int32, [None, None])
                input_emb = tf.nn.embedding_lookup(word_embeddings, input_x)

        with tf.variable_scope('first_lstm'):
            cell_1 = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
            dropout_cell_1 = tf.contrib.rnn.DropoutWrapper(cell_1, output_keep_prob=self.keep_prob)
            multi_dropout_cell_1 = [dropout_cell_1 for _ in range(self.num_layers)]
            multi_lstm_cell_1 = tf.contrib.rnn.MultiRNNCell(multi_dropout_cell_1, state_is_tuple=True)
            first_outputs, _ = tf.nn.dynamic_rnn(cell=multi_lstm_cell_1, inputs=input_emb, sequence_length=input_seqlen, dtype=tf.float32)

        if self.add_keyword_attention:
            keyword_emb = tf.nn.embedding_lookup(word_embeddings, keyword_id)
            with tf.variable_scope('first_attention'):
                atten_weight_1 = tf.get_variable('atten_weight_1',[self.word_emb_dim, self.hidden_dim],
                                               initializer=tf.truncated_normal_initializer(stddev=0.01),
                                               dtype=tf.float32)
                trans_keyword_emb = tf.matmul(keyword_emb, atten_weight_1)
                print(trans_keyword_emb.shape)  # (100, 100)
                atten_dot = tf.tensordot(first_outputs, tf.transpose(trans_keyword_emb), axes=1)
                alphas = tf.nn.softmax(atten_dot)
                print(alphas.shape)  # (?, 500,100)
                atten_outputs = tf.reduce_sum(tf.expand_dims(alphas, -1) * trans_keyword_emb, 2)
                print(atten_outputs.shape)  #(?, 500, 100)

            with tf.variable_scope('concate'):
                concate_inputs = tf.concat([input_emb, atten_outputs], 2)
                print(concate_inputs.shape) #(?, 500, 200)

            with tf.variable_scope('second_lstm'):
                cell_2 = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim, state_is_tuple=True)
                dropout_cell_2 = tf.contrib.rnn.DropoutWrapper(cell_2, output_keep_prob=self.keep_prob)
                multi_dropout_cell_2 = [dropout_cell_2 for _ in range(self.num_layers)]
                multi_lstm_cell_2 = tf.contrib.rnn.MultiRNNCell(multi_dropout_cell_2, state_is_tuple=True)
                second_outputs, _ = tf.nn.dynamic_rnn(cell=multi_lstm_cell_2, inputs=concate_inputs, sequence_length=input_seqlen, dtype=tf.float32)

        else:
            second_outputs = first_outputs
        print('sencond_outputs shape: ', second_outputs.shape)

        if self.add_second_attention:
            with tf.variable_scope('second_attention'):
                second_atten_vect = tf.get_variable('second_atten_w', [self.hidden_dim],
                                               initializer=tf.truncated_normal_initializer(stddev=0.01), dtype=tf.float32)
                atten_dot = tf.tensordot(first_outputs, tf.transpose(second_atten_vect), axes=1)
                alphas = tf.nn.softmax(atten_dot)
                print('alphas shape: ',alphas.shape)
                last_outputs = tf.reduce_sum(tf.expand_dims(alphas, -1) * second_outputs, 1)

        else:
            last_outputs = tf.reduce_mean(second_outputs, axis=1)
        print('last_outputs shape: ', last_outputs.shape)

        with tf.variable_scope('full_connect_layer'):
            # fc_1 = tf.layers.dense(last, self.hidden_dim,activation=tf.nn.relu)
            logit = tf.layers.dense(last_outputs, self.num_classes)
            y_pred = tf.argmax(tf.nn.softmax(logit), 1)

        with tf.variable_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=input_y)
            loss = tf.reduce_mean(cross_entropy)
            optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

        with tf.variable_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(input_y, 1), y_pred)
            acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for t in range(num_epochs):
                for i in range(int(len(train_x) / batch_size)+1):
                    k = batch_size * i
                    h = k + batch_size
                    assert train_x[k:h] != [], 'empty input!'
                    batch_x, batch_seqlen = self.batch_seq_padding(train_x[k:h])
                    batch_y = train_y[k:h]
                    train_pred_y, train_loss, _ = sess.run([y_pred, loss, optim], feed_dict={input_x: batch_x, input_y:batch_y ,
                                                                                             input_seqlen: batch_seqlen})
                    if i % 100 == 0:
                        print('batch {}, training loss = {}'.format(i, train_loss))
                    if i % 500 == 0:
                        test_acc = []
                        test_loss = []
                        for j in range(int(len(test_x) / batch_size)):
                            m = batch_size * j
                            n = m + batch_size
                            assert test_x[m:n] != [], 'empty input!'
                            batch_test_x, batch_test_seqlen = self.batch_seq_padding(test_x[m:n])
                            batch_test_y = test_y[m:n]
                            t_loss, accuracy, test_pred_y = sess.run([loss, acc, y_pred], feed_dict={input_x: batch_test_x, input_y: batch_test_y,
                                                                input_seqlen: batch_test_seqlen})
                            test_acc.append(accuracy)
                            test_loss.append(t_loss)
                        print('   test loss = {}, test accuracy is {}'.format(np.mean(test_loss), np.mean(test_acc)))
                        print('-------------------------------------------------------------------')
            print('training done!')
            print('testing ...')
            test_acc = []
            for j in range(int(len(test_x) / batch_size)):
                m = batch_size * j
                n = m + batch_size
                assert test_x[m:n] != [], 'empty input!'
                batch_test_x, batch_test_seqlen = self.batch_seq_padding(test_x[m:n])
                batch_test_y = test_y[m:n]
                accuracy = sess.run([acc], feed_dict={input_x: batch_test_x, input_y: batch_test_y, input_seqlen: batch_test_seqlen})
                test_acc.append(accuracy)
            print('the final test accuracy is {}'.format(np.mean(test_acc)))
            print('done!')
