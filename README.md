# ICD_10-text-classification
This code is for medical home page ICD-10 main diagnose code classification base on electronic medical record.

## data description
Because the whole data is private, it is not available for public. We offer the sample data, which is seen in sample_data.csv
The data includes 7 feature text data (chief complaint, present illness history, examination report, first progress note, operation record, whole progress note, discharge records) and the label(ICD-10 diagnose code). All the chinese text data have been tokenized by jieba tokenizer.
## load_data.py
Code for loading and preprocessing data.
## model.py
Code for the model. The model structure is shown in the below picture.
![picture](https://github.com/zhanghk-pku/ICD_10-text-classification/blob/master/picture.png)
## runner.py
Code for training and testing model. 
```
python runner.py
```
