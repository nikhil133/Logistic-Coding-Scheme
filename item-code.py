import pickle
import pandas as pd 
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score

nltk.download('punkt')
dataset=pd.read_csv("item-codes-sample.csv")
def DataPreprocessing():
    dataset['items']=dataset['items'].replace('\W',' ',regex=True)
    dataset['items']=dataset['items'].replace('\d',' ',regex=True)
    dataset['items'] = dataset['items'].str.lower()
    return dataset['items'].to_numpy(), dataset['class_code'].to_numpy()
X,y=DataPreprocessing()
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,y,test_size=0.2,random_state=42)
vector=CountVectorizer()
train_v=vector.fit_transform(Xtrain)
tfid=TfidfTransformer()
train_tfid=tfid.fit_transform(train_v)
classifier=MultinomialNB().fit(train_tfid,Ytrain)
with open('count_vector.pkl','wb')as cvid:
    pickle.dump(vector,cvid)
with open('tfid_vector.pkl','wb')as tfvid:
    pickle.dump(tfid,tfvid)
with open('saved_classifier.pkl','wb')as fid:
    pickle.dump(classifier,fid)

test_v=vector.transform(Xtest)
test_tfid=tfid.transform(test_v)
predicted=classifier.predict(test_tfid)
print("Accuracy ",accuracy_score(Ytest,predicted))
'''
#vector.get_feature_names()
##print(vector.get_feature_names())
#counts=vector.transform(text)
##print(counts.toarray())    

value=map(str.lower,["HP Printer and Scanner"])
print(value)
test_v=vector.transform(value)
print("test_v ",test_v)
test_tfid=tfid.transform(test_v)
predicted=classifier.predict(test_tfid)
print(predicted[0])
category=Categorise(predicted[0])
print(category)
'''