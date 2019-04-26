import pandas as pd
import numpy as np
import re
from itertools import chain
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
        stop_words = ['非pvc', '氯化钠注射液', '葡萄糖注射液','患者','正常','明显','每次','入院','出院','年','月','日','天',
                      '均匀','未见异常','显示','对称','符合','我科','我院','外院','尚可','年余','月余','体重','实验室','可见',
                      '一般','进一步','大小','明确','表现','提示','mg','mmol','aa','给予','之后','检查','发现','治疗','少量',
                      '术后','考虑','ml','cm','诊断']
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

    def creat_id_sentences(self, data, label, word_vocab, feature_vocab):
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
        labels = []
        for i in range(len(data)):
            sentence = []
            for feature in feature_sentences:
                sentence += feature[i]
            if sentence != []:
                merge_feature_sentences.append(sentence)
                labels.append(label[i])
        return merge_feature_sentences, labels

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
            key_words = []
            word_tfidf_dict = dict(zip(words, weight[i]))
            sorted_dict = sorted(word_tfidf_dict.items(), key=lambda x: x[1], reverse=True)
            for item in sorted_dict[:keyword_num]:
                key_words.append(item[0])
            total_key_words.append(key_words)
            label_key_words[list(label_text.keys())[i]] = ' '.join(key_words)
        return total_key_words, label_key_words
