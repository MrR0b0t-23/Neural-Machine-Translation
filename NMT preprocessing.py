import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

tamil_dir = []
eng_dir = []
for i in range(1,7):
    tamil_dir.append('tam'+str(i)+".txt")
    eng_dir.append('eng'+str(i)+".txt")

eng_corpus = []
for file in eng_dir:
    path = "C:\\Users\\admin\\Desktop\\Machine Translation\\Tamil-English-Dataset-master\\Dataset\\"+file
    temp = pd.read_csv(path, delimiter = '\t', header =None, engine ='python-fwf',encoding='utf8')
    eng_corpus.append(temp[0].values)

eng_corpus = [j for sub in eng_corpus for j in sub]
eng_corpus = np.array(eng_corpus)
print(eng_corpus.shape)

tamil_corpus = []
for file in tamil_dir:
    path = "C:\\Users\\admin\\Desktop\\Machine Translation\\Tamil-English-Dataset-master\\Dataset\\"+file
    temp = pd.read_csv(path, delimiter = '\t', header =None,engine ='python-fwf',encoding='utf8')
    tamil_corpus.append(temp[0].values)

tamil_corpus = [j for sub in tamil_corpus for j in sub]
tamil_corpus = np.array(tamil_corpus)
print(tamil_corpus.shape)

df = pd.DataFrame()
df['english'] = eng_corpus
df['tamil'] = tamil_corpus

train, test = train_test_split(df, test_size=0.35, random_state=23)
test, val = train_test_split(test, test_size=0.15, random_state=23)
train.to_csv("tamil-eng-train.csv", index= False)
test.to_csv("tamil-eng-test.csv", index= False)
val.to_csv("tamil-eng-val.csv", index= False)