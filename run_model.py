import numpy as np
from itertools import chain
from data_process import DataProcess
from model import Models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


seq_length = 500        # 序列长度
keyword_num = 10         # 关键词（特征词）个数
hidden_dim = 100         # LSTM的隐层神经元个数（输出维度）
word_emb_dim = 50       # 词向量维度
feature_emb_dim = 50    # 特征名称embedding维度
keep_prob = 0.8          # dropout保留比例
num_layers = 1           # LSTM层数
batch_size = 50          # 每个batch的大小
learning_rate = 0.001    # 学习率
num_epochs = 2          # 训练数据迭代次数
add_feature = True       # 是否加特征名embedding
add_keyword_attention = True      # 是否加关键词attention

print('loading and processing data ...')
process = DataProcess(add_feature, add_keyword_attention)
train_data,train_label = process.load_data('train_data.csv')
test_data,test_label = process.load_data('test_data.csv')
merge_data = process.merge_feature_text(train_data)

word_vocab, feature_vocab = process.build_vocab(merge_data)
vocab_size = len(word_vocab)
print('the vocabulary size is {}'.format(vocab_size))

features = process.feature_names
num_features = len(features)
num_classes = len(set(train_label))
test_classes = len(set(test_label))
print('num_classes = {}, num_features = {}'.format(num_classes,num_features))
print('test num_classes = {}'.format(test_classes))

train_sentences = process.creat_id_sentences(train_data, word_vocab, feature_vocab)
test_sentences = process.creat_id_sentences(test_data, word_vocab, feature_vocab)

sentences_length = [len(sentence) for sentence in train_sentences]
mean_seq_length = np.mean(sentences_length)
max_seq_length = np.max(sentences_length)
print('mean_seq_length = {}, max_seq_length = {}'.format(mean_seq_length, max_seq_length))

if add_keyword_attention:
    keywords, label_key_words = process.extrac_keywords(train_data, train_label, keyword_num)
    for key, value in label_key_words.items():
        print(key, ':', value)
    keywords_id = [word_vocab[word] for word in list(chain(*keywords))]
else:
    keywords_id = None

train_x, mask_train = process.data_padding(train_sentences, seq_length, hidden_dim)
train_y = process.label_to_onehot(train_label)
test_x, mask_test = process.data_padding(test_sentences, seq_length, hidden_dim)
test_y = process.label_to_onehot(test_label)

print('train size = {}, test size = {}'.format(len(train_x),len(test_x)))


print('loading model ...')
models = Models(vocab_size, num_classes, word_emb_dim, feature_emb_dim, add_feature, seq_length,
                 hidden_dim, num_features, keep_prob, add_keyword_attention)
print('start training model ...')
models.lstm_model(train_x, train_y, test_x, test_y, keywords_id, mask_train, mask_test,
                   num_layers, batch_size, learning_rate, num_epochs)
