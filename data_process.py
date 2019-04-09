import pandas as pd
import numpy as np
import re
from itertools import chain
import heapq
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class DataProcess(object):
    def __init__(self, add_feature, add_keyword_attention):
        self.add_feature = add_feature
        self.add_keyword_attention = add_keyword_attention
        self.feature_names = ['主诉', '现病史', '检查', '首次病程记录', '手术记录', '查房记录', '出院记录']
        self.label_name = ['编码']

    def load_data(self,data_file):
        raw_data = pd.read_csv(data_file, names=self.feature_names + self.label_name, header=None)
        data = raw_data[self.feature_names]
        label = list(raw_data['编码'])
        return data, label


    def clean_str(self,text_list):
        clean_sentences = []
        stop_words = ['患者','检查','治疗','正常','明显','入院','于','示','出院','mg','ml','mmol','cm','双侧','左侧','右侧','给予']
        for sentence in text_list:
            sentence = str(sentence).replace('nan','').lower()
            sentence = re.sub('[／\d\-，。：“”℃%、,+？；/（）()\.\r\n:×^‘’\']',' ', sentence)
            for word in stop_words:
                sentence = sentence.replace(word,' ')
            clean_sentences.append(sentence)
        return clean_sentences

    def merge_feature_text(self, data):
        feature_text = []
        for name in self.feature_names:
            feature_text.append(list(data[name]))
        merge_sentences = []
        for i in range(len(data)):
            sentence = []
            for text in feature_text:
                sentence.append(str(text[i]))
            sentence = ' '.join(sentence)
            merge_sentences.append(sentence)
        merge_data = self.clean_str(merge_sentences)
        return merge_data

    def build_vocab(self, merge_data):
        split_sentences = [sentence.split() for sentence in merge_data]
        total_words = list(set(chain(*split_sentences)))
        word_vocab = {}
        feature_vocab = {}
        for i, word in enumerate(total_words,1):
            word_vocab[word] = i
        for i, name in enumerate(self.feature_names, 1):
            feature_vocab[name] = i
        return word_vocab, feature_vocab

    def creat_id_sentences(self, data, word_vocab, feature_vocab):
        feature_sentences = []
        for name in self.feature_names:
            feature_text = self.clean_str(list(data[name]))
            sentences = [sentence.split() for sentence in feature_text]
            id_sentences = []
            for sentence in sentences:
                id_sentence = []
                for word in sentence:
                    if word in word_vocab:
                        word_id = word_vocab[word]
                    else:
                        word_id = 0
                    if self.add_feature:
                        id_sentence.append([word_id,feature_vocab[name]])
                    else:
                        id_sentence.append(word_id)
                id_sentences.append(id_sentence)
            feature_sentences.append(id_sentences)
        merge_feature_sentences = []
        for i in range(len(data)):
            sentence = []
            for feature in feature_sentences:
                sentence += feature[i]
            merge_feature_sentences.append(sentence)
        return merge_feature_sentences


    def data_padding(self, text_data, seq_length, hidden_dim):
        x_padding = []
        mask_matrix = np.ones([len(text_data),seq_length,hidden_dim])
        print('padding data and generate mask matrix ...')
        for i in range(len(text_data)):
            sentence = text_data[i]
            if len(sentence) < seq_length:
                mask_matrix[i][len(sentence):] = 0
                if self.add_feature:
                    sentence = sentence + [[0, 0]] * (seq_length - len(sentence))
                else:
                    sentence = sentence + [0] * (seq_length - len(sentence))
            else:
                sentence = sentence[:seq_length]
            x_padding.append(sentence)

        return np.array(x_padding), mask_matrix

    def label_to_onehot(self, label):
        one_hot_label = pd.get_dummies(label)
        return np.array(one_hot_label)

    def extrac_keywords(self, data, label, keyword_num):
        print('extract key words ...')
        merge_data = self.merge_feature_text(data)
        label_text = {}
        for i in range(len(label)):
            if label[i] not in label_text:
                label_text[label[i]] = [merge_data[i]]
            else:
                label_text[label[i]].append(merge_data[i])
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
        label_key_words = {}
        for i in range(len(weight)):
            keyword_indexs = map(list(weight[i]).index, heapq.nlargest(keyword_num, list(weight[i])))
            key_words = [words[j] for j in keyword_indexs]
            total_key_words.append(key_words)
            label_key_words[list(label_text.keys())[i]] = ' '.join(key_words)
        return total_key_words, label_key_words



