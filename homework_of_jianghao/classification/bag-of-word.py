import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
row=5130
'''
def dict():
    file1=open("train.tsv","rb")
    file2=open("file_out.txt","w",encoding='utf-8')
    dict={}
    en = ("abcdefghijklmnopqrstuvwxyz")
    x=0
    for line in file1:
        content=line.decode('utf-8').split()
        for i in range(len(content)):
            content[i] = content[i].lower()
            if(content[i][0] in en):
                content[i] = lemmatizer.lemmatize(content[i], pos='n')
                content[i] = lemmatizer.lemmatize(content[i], pos='v')
                content[i] = lemmatizer.lemmatize(content[i], pos='a')
                content[i] = lemmatizer.lemmatize(content[i], pos='r')
                if(content[i] in dict):
                    dict[content[i]]=dict[content[i]]+1
                else:
                    dict[content[i]]=1
                    x=x+1
    dic = sorted(dict.items(), key=lambda d: d[1], reverse=True)
    for i in range(len(dic)):
        file2.write("".join(list(dic[i][0])))
        file2.write(" ")
        file2.write(str(dic[i][1]))
        file2.write("\n")
        
    file1.close()
    file2.close()
    return x
print(dict())
'''
def get_dict():
    file=open("file_out.txt","rb")
    i=0
    dict={}
    for line in file:
        l=line.decode('utf-8').split()
        dict[l[0]]=i
        i=i+1
    file.close()
    return dict

def sen2vec(sen,dict):
    sen_zu=sen.lower().split()

    vector=np.zeros((1,row))
    for i in range(len(sen_zu)):
        sen_zu[i] = lemmatizer.lemmatize(sen_zu[i], pos='v')
        sen_zu[i] = lemmatizer.lemmatize(sen_zu[i], pos='n')
        sen_zu[i] = lemmatizer.lemmatize(sen_zu[i], pos='a')
        sen_zu[i] = lemmatizer.lemmatize(sen_zu[i], pos='r')
        if(sen_zu[i] in dict):
            vector[0][dict[sen_zu[i]]]=vector[0][dict[sen_zu[i]]]+1
    vector[0][row-1]=1
    return vector



i=0
w=np.full([row,5],0.0001,dtype='float32')
learning_rate=0.11

right=0
all=0
_=0

'''
for m in range(100):
    f2 = open("train_data_reduced.txt", "rb")
    for line in f2:
        i+=1
        con = line.decode('utf-8').split('\t')
        xxx=str(con[2]).split()
        xx=xxx[-1]
        sen_vector = sen2vec(str(con[2]), get_dict())
        tar = np.zeros([1, 5])
        tar[0][int(xx)] = 1
        y_ = np.e ** (np.matmul(sen_vector, w))
        y1 = y_ / np.sum(y_)
        y = tar-y1
        w += learning_rate * np.dot(np.transpose(sen_vector), y)
        cross = -np.sum(np.log(y1) * tar)
        if(i%200==0):
            print(i, cross)
    f2.close()
f1=open("test_data_reduced.txt","rb")
for line in f1:
    con = line.decode('utf-8').split('\t')
    xxx = str(con[2]).split()
    xx = xxx[-1]
    sen_vector = sen2vec(str(con[2]), get_dict())
    tar = np.zeros([1, 5])
    tar[0][int(xx)] = 1
    y_ = np.e ** (np.matmul(sen_vector, w))
    y = y_ / np.sum(y_)
    lab = np.argmax(y)
    if (lab == int(xx)):
        right += 1
    all += 1
print(right/all)
f1.close()

'''
dd=get_dict()
f=open("train.tsv","rb")
for line in f:
    if(_==0):
        _=1
    else:
        con=line.decode('utf-8').split('\t')
        sen_vector=sen2vec(str(con[2]),dd)
        tar=np.zeros([1,5])
        tar[0][int(con[3])]=1
        if(i<156061*3/4):
            i+=1
            y_ = np.e ** (np.matmul(sen_vector, w))
            y1 = y_ / np.sum(y_)
            y=tar-y1
            w+=learning_rate*np.dot(np.transpose(sen_vector),y)
            cross=-np.sum(np.log(y1)*tar)
            if(i%300==0):
                print(i,cross)
        else:
            y_ = np.e ** (np.matmul(sen_vector, w))
            y = y_ / np.sum(y_)
            lab=np.argmax(y)
            if(lab==int(con[3])):
                right+=1
            all+=1
print(right/all)
f.close()
