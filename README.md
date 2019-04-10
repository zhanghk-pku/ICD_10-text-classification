# ICD_10-text-classification
This code is for medical home page ICD-10 main diagnose code classification base on electronic medical record.

## data description
Because the whole data is private, it is not available for public. We offer the sample data, which is seen in sample_data.csv
The data includes 7 feature text data (chief complaint, present illness history, examination report, first progress note, operation record, ward-round records, discharge records) and the label(ICD-10 diagnose code). All the chinese text data have been tokenized by jieba tokenizer.
## data_split.py
Code for dividing the whole data into three dataset: train data, validation data, test data. 
## data_process.py
Code for data process, including clean text, build vocabulary, transform sentences into word index representation and extract the topic word(key words for each classes).
## model.py
Code for the model. The model structure is shown in the below picture.
![picture](https://github.com/zhanghk-pku/ICD_10-text-classification/blob/master/picture.png)
## run_model.py
Code for training and testing model. just use the command:
```
python run_model.py
```
## hyper-parameter explanation
####  1. data split hyper-parameter
```
filter_num = 100        # 数据过滤参数，滤掉样本数小于filter_num的类
valid_rate = 0.1        # 验证集比例
test_rate = 0.1         # 测试集比例
```

####  2. model hyper-parameter
```
seq_length = 1000                 # 序列长度
keyword_num = 10                  # 关键词（特征词）个数
test_rate = 0.1                   # 测试集比例
hidden_dim = 100                  # LSTM的隐层神经元个数（输出维度）
word_emb_dim = 200                # 词向量维度
feature_emb_dim = 100             # 特征名称embedding维度
keep_prob = 0.8                   # dropout保留比例
num_layers = 1                    # LSTM层数
batch_size = 20                   # 每个batch的大小
learning_rate = 0.001             # 学习率
num_epochs = 10                   # 训练数据迭代次数
add_feature = True                # 是否加特征名embedding
add_keyword_attention = True      # 是否加关键词attention
```
