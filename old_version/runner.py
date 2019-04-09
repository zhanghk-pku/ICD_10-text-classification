import numpy as np
from itertools import chain
from load_data import LoadData
from model import Models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

# 设置超参数
filter_num = 2           # 数据过滤参数，滤掉样本数小于filter_num的类
seq_length = 1000        # 序列长度
keyword_num = 10         # 关键词（特征词）个数
test_rate = 0.1          # 测试集比例
hidden_dim = 100         # LSTM的隐层神经元个数（输出维度）
word_emb_dim = 200       # 词向量维度
feature_emb_dim = 100    # 特征名称embedding维度
keep_prob = 0.8          # dropout保留比例
num_layers = 1           # LSTM层数
batch_size = 20          # 每个batch的大小
learning_rate = 0.001    # 学习率
num_epochs = 10          # 训练数据迭代次数
add_feature = True       # 是否加特征名embedding
add_keyword_attention = True      # 是否加关键词attention

print('loading data ...')
text_data = LoadData('sample_data.csv', filter_num, add_feature)
word_vocab, feature_vocab = text_data.build_vocab()
vocab_size = len(word_vocab)
print('the vocabulary size is {}'.format(vocab_size))
label = text_data.label

features = text_data.feature_names
num_features = len(features)
num_classes = len(set(label))
print('num_classes = {}, num_features = {}'.format(num_classes,num_features))

sentences = text_data.creat_id_sentences(word_vocab, feature_vocab)
sentences_length = [len(sentence) for sentence in sentences]
mean_seq_length = np.mean(sentences_length)
max_seq_length = np.max(sentences_length)
print('mean_seq_length = {}, max_seq_length = {}'.format(mean_seq_length, max_seq_length))

keywords = text_data.extrac_keywords(keyword_num)
keywords_id = [word_vocab[word] for word in list(chain(*keywords))]

train_x, train_y, test_x, test_y, mask_train, mask_test = text_data.data_split(text_data=sentences, seq_length=seq_length,
                                                                               hidden_dim=hidden_dim,test_rate = test_rate)

train_x = np.array(train_x * num_epochs)
train_y = np.array(train_y * num_epochs)
mask_train = np.array(mask_train * num_epochs)
test_x = np.array(test_x)
test_y = np.array(test_y)
mask_test = np.array(mask_test)

print('loading model ...')
models = Models(vocab_size, num_classes, word_emb_dim, feature_emb_dim, add_feature, seq_length,
                 hidden_dim, num_features, keep_prob, add_keyword_attention)
print('start training model ...')
models.lstm_model(train_x, train_y, test_x, test_y, keywords_id, mask_train, mask_test,
                   num_layers, batch_size, learning_rate)
