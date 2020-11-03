import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

parent_cat={"01":"Machine","02":"Micro Electronics","04":"Job Work","03":"Network Items","10":"Weapons","52":"Measuring Tools","23":"Motor Vehicle Trailer and Cycles"}
sub_class_cat={"0101":"Electronics","0201":"Electronics","0202":"Consumable","0404":"Shifting","0104":"Communication System IT","5220":"Inspection Guage & Precision Layout Tool's","2310":"Passenger Motor Vehicle","1000":"Weapon"}
sub_group_cat={"010201":"Scanning","020202":"Fibre","040404":"Stores","010401":"Computer","522001":"Dynamo Meter","231001":"Passenger Vehicle","231002":"Car or Jeep","100000":"Weapon"}

def Categorise(key):
    if len(str(int(key/10000)))==1:
        parent_key='0'+str(int(key/10000))
    else:
        parent_key=str(int(key/10000))
    if len(str(int(key/100)))==3:
        sub_key='0'+str(key%10000)
    else:
        sub_key=str(int(key/100))
    if len(str(key))==5:
        class_key='0'+str(key)
    else:
        class_key=str(key)
    print([parent_cat[parent_key],sub_class_cat[sub_key],sub_group_cat[class_key]])
    try:
        return [parent_cat[parent_key],sub_class_cat[sub_key],sub_group_cat[class_key]]
    except Exception:
        return ["Unknown","Unknown","Unknown"]
    
value=[input("Enter product description ")]
value=map(str.lower,value)

with open('count_vector.pkl','rb')as vfid:
    vector=pickle.load(vfid)

with open('tfid_vector.pkl','rb')as tfvid:
    tfid=pickle.load(tfvid)

with open('saved_classifier.pkl','rb')as fid:
    classifier=pickle.load(fid)


train_v=vector.transform(value)
train_tfid=tfid.transform(train_v)
predicted=classifier.predict(train_tfid)
classes=Categorise(predicted[0])
print(predicted)
print("Master category {} \nSub Class {} \nSub Group {}".format(classes[0],classes[1],classes[2]))
'''
category=Categorise(predicted[0])
print(category)
'''