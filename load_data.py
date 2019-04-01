import pandas as pd
import numpy as np
import random
import re
from itertools import chain
import heapq
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

class LoadData(object):
    def __init__(self,data_file, filter_num, add_feature):
        self.data_file = data_file
        self.filter_num = filter_num
        self.add_feature = add_feature
        self.feature_names = ['主诉', '现病史', '检查', '首次病程记录', '手术记录', '查房记录', '出院记录']
        self.label_name = ['编码']
        raw_data = pd.read_csv(self.data_file, names=self.feature_names + self.label_name)
        dropna_data = raw_data.dropna(axis=0, how='all', subset=self.feature_names)
        self.filter_data = self.data_filter(dropna_data)
        self.label = list(self.filter_data['编码'])


    def data_filter(self, data):
        ICD_dict = dict(data["编码"].value_counts())
        reserved_ICD = []
        for code in ICD_dict.keys():
            if ICD_dict[code] >= self.filter_num:
                reserved_ICD.append(code)
        droped_index = []
        for i,code in enumerate(data["编码"]):
            if code not in reserved_ICD:
                droped_index.append(i)
        data = data.drop(labels=droped_index, axis=0).reset_index(drop=True)
        return data

    def clean_str(self,text_list):
        return [re.sub('[／\d\-，。：“”℃%、,+？；/（）()\.\r\n:×^‘’\']',' ',str(sentence).replace('nan','').lower())
                for sentence in text_list]

    def merge_feature_text(self):
        feature_text = []
        for name in self.feature_names:
            feature_text.append(list(self.filter_data[name]))
        merge_sentences = []
        for i in range(len(self.filter_data)):
            sentence = []
            for text in feature_text:
                sentence.append(str(text[i]))
            sentence = ' '.join(sentence)
            merge_sentences.append(sentence)
        merge_data = self.clean_str(merge_sentences)
        return merge_data

    def build_vocab(self):
        merge_data = self.merge_feature_text()
        split_sentences = [sentence.split() for sentence in merge_data]
        total_words = list(set(chain(*split_sentences)))
        word_vocab = {}
        feature_vocab = {}
        for i, word in enumerate(total_words,1):
            word_vocab[word] = i
        for i, name in enumerate(self.feature_names,1):
            feature_vocab[name] = i
        return word_vocab, feature_vocab

    def creat_id_sentences(self, word_vocab, feature_vocab):
        feature_sentences = []
        for name in self.feature_names:
            feature_text = self.clean_str(list(self.filter_data[name]))
            sentences = [sentence.split() for sentence in feature_text]
            if self.add_feature:
                id_sentences = [[(word_vocab[word], feature_vocab[name]) for word in sentence]
                                for sentence in sentences]
            else:
                id_sentences = [[word_vocab[word] for word in sentence] for sentence in sentences]
            feature_sentences.append(id_sentences)
        merge_feature_sentences = []
        for i in range(len(self.filter_data)):
            sentence = []
            for feature in feature_sentences:
                sentence += feature[i]
            merge_feature_sentences.append(sentence)
        return merge_feature_sentences


    def data_split(self, text_data, seq_length, hidden_dim, test_rate):
        x_padding = []
        mask_matrix = []
        mask_one = [1.] * hidden_dim
        mask_zero = [0.] * hidden_dim
        print('padding data and generate mask matrix ...')
        for sentence in text_data:
            if len(sentence) < seq_length:
                mask_value = [mask_one] * len(sentence) + [mask_zero] * (seq_length - len(sentence))
                if self.add_feature:
                    sentence = sentence + [(0, 0)] * (seq_length - len(sentence))
                else:
                    sentence = sentence + [0] * (seq_length - len(sentence))
            else:
                mask_value = [mask_one] * seq_length
                sentence = sentence[:seq_length]
            x_padding.append(sentence)
            mask_matrix.append(mask_value)
        one_hot_label = pd.get_dummies(self.filter_data['编码'])
        y_one_hot = np.array(one_hot_label).tolist()

        print('split train-test data ...')
        test_size = int(len(x_padding) * test_rate)
        test_id = random.sample(range(0, len(x_padding)), test_size)
        test_x = [x_padding[i] for i in test_id]
        test_y = [y_one_hot[i] for i in test_id]
        mask_test = [mask_matrix[i] for i in test_id]
        for i in range(len(test_id)):
            x_padding.remove(test_x[i])
            y_one_hot.remove(test_y[i])
            mask_matrix.remove(mask_test[i])

        return x_padding, y_one_hot, test_x, test_y, mask_matrix, mask_test

    def extrac_keywords(self, keyword_num):
        print('extract key words ...')
        merge_data = self.merge_feature_text()
        label_text = {}
        for i in range(len(self.label)):
            if self.label[i] not in label_text:
                label_text[self.label[i]] = [merge_data[i]]
            else:
                label_text[self.label[i]].append(merge_data[i])

        corpus = []
        for key, value in label_text.items():
            corpus.append(' '.join(value))

        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        words = vectorizer.get_feature_names()
        print('tf-idf words size is {}'.format(len(words)))
        weight = tfidf.toarray()
        total_key_words = []
        for i in range(len(weight)):
            keyword_indexs = map(list(weight[i]).index, heapq.nlargest(keyword_num, list(weight[i])))
            key_words = [words[i] for i in keyword_indexs]
            total_key_words.append(key_words)
        return total_key_words



