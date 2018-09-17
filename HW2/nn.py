import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import  Dense
from random import shuffle
from numpy.linalg import inv
import matplotlib.pyplot as plt
from math import floor, log
import os

output_dir = "output/"

def dataProcess_X(rawData):

    #sex 只有两个属性 先drop之后处理
    if "income" in rawData.columns:
        Data = rawData.drop(["sex", 'income'], axis=1)
    else:
        Data = rawData.drop(["sex"], axis=1)
    listObjectColumn = [col for col in Data.columns if Data[col].dtypes == "object"] #读取非数字的column
    listNonObjedtColumn = [x for x in list(Data) if x not in listObjectColumn] #数字的column

    ObjectData = Data[listObjectColumn]
    NonObjectData = Data[listNonObjedtColumn]
    #insert set into nonobject data with male = 0 and female = 1
    NonObjectData.insert(0 ,"sex", (rawData["sex"] == " Female").astype(np.int))
    #set every element in object rows as an attribute
    ObjectData = pd.get_dummies(ObjectData)

    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    Data_x = Data.astype("int64")
    # Data_y = (rawData["income"] == " <=50K").astype(np.int)

    #normalize
    Data_x = (Data_x - Data_x.mean()) / Data_x.std()

    return Data_x

def dataProcess_Y(rawData):
    df_y = rawData['income']
    Data_y = pd.DataFrame((df_y==' >50K').astype("int64"), columns=["income"])
    return Data_y

if __name__ == "__main__":
    trainData = pd.read_csv("data/train.csv")
    testData = pd.read_csv("data/test.csv")
    ans = pd.read_csv("data/correct_answer.csv")

    # here is one more attribute in trainData
    x_train = dataProcess_X(trainData).drop(['native_country_ Holand-Netherlands'], axis=1).values
    x_test = dataProcess_X(testData).values
    y_train = dataProcess_Y(trainData).values
    y_ans = ans['label'].values

    model = Sequential()

    model.add(Dense(units=600, activation='sigmoid', input_dim=106))
    model.add(Dense(units=600, activation='sigmoid'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=50)
    score = model.evaluate(x_test, y_ans)
    result = np.squeeze(model.predict(x_test))
    # print('Total loss on Testing set: ', score[0])
    # print('Accuracy of Testing set: ', score[1])


    model.save(os.path.join(output_dir+'nn_model.h5'))

    y_ = np.around(result).astype(np.int)
    df = pd.DataFrame({"id": np.arange(1, 16282), "label": y_})
    result = (np.squeeze(y_ans) == y_)
    print('Test acc = %f' % (float(result.sum()) / result.shape[0]))
    df = pd.DataFrame({"id": np.arange(1, 16282), "label": y_})
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df.to_csv(os.path.join(output_dir + 'nn_output.csv'), sep='\t', index=False)
