import numpy as np
from itertools import chain
from data_process import DataProcess
from model import Models
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


keyword_num = 10         # 关键词（特征词）个数
hidden_dim = 100         # LSTM的隐层神经元个数（输出维度）
word_emb_dim = 100       # 词向量维度
feature_emb_dim = 50     # 特征名称embedding维度
keep_prob = 0.8          # dropout保留比例
num_layers = 1           # LSTM层数
batch_size = 50          # 每个batch的大小
learning_rate = 0.01     # 学习率
num_epochs = 2           # 训练数据迭代次数
add_feature_emb = True            # 是否加特征名embedding
add_keyword_attention = True      # 是否加关键词attention
add_second_attention = True       # 是否加第二层的attention

print('loading and processing data ...')
process = DataProcess(add_feature_emb, add_keyword_attention)
train_data,train_label = process.load_data('train_data_10.csv')
test_data,test_label = process.load_data('test_data_10.csv')
merge_data = process.merge_feature_text(train_data)

word_vocab, feature_vocab = process.build_vocab(merge_data)
vocab_size = len(word_vocab)
print('the vocabulary size is {}'.format(vocab_size))

train_sentences, train_labels = process.creat_id_sentences(train_data, train_label, word_vocab, feature_vocab)
test_sentences, test_labels = process.creat_id_sentences(test_data, test_label, word_vocab, feature_vocab)

#
features = process.feature_names
num_features = len(features)
num_classes = len(set(train_label))
test_classes = len(set(test_label))
print('num_classes = {}, num_features = {}'.format(num_classes,num_features))
print('test num_classes = {}'.format(test_classes))

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


train_y = process.label_to_onehot(train_labels)
test_y = process.label_to_onehot(test_labels)


print('train size = {}, test size = {}'.format(len(train_sentences),len(test_sentences)))

print('loading model ...')
models = Models(vocab_size, num_classes, num_features, word_emb_dim, feature_emb_dim, hidden_dim, num_layers,
                 learning_rate, keep_prob, add_feature_emb, add_second_attention, add_keyword_attention)
print('start training model ...')
models.lstm_model(train_sentences, train_y, test_sentences, test_y, keywords_id, batch_size, num_epochs)
