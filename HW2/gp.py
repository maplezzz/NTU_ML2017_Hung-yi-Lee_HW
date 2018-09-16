import pandas as pd
import numpy as np
from random import shuffle
from numpy.linalg import inv
from math import floor, log
import os
import argparse



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


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, (1-(1e-8)))

def _shuffle(X, Y):                                 #X and Y are np.array
    randomize = np.arange(X.shape[0])
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def split_valid_set(X, Y, percentage):
    all_size = X.shape[0]
    valid_size = int(floor(all_size * percentage))

    X, Y = _shuffle(X, Y)
    X_valid, Y_valid = X[ : valid_size], Y[ : valid_size]
    X_train, Y_train = X[valid_size:], Y[valid_size:]

    return X_train, Y_train, X_valid, Y_valid

def valid(X, Y, mu1, mu2, shared_sigma, N1, N2):
    sigma_inv = inv(shared_sigma)
    w = np.dot((mu1-mu2), sigma_inv)
    X_t = X.T
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(float(N1)/N2)
    a = np.dot(w,X_t) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y) == y_)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return

def train(X_train, Y_train):
    # vaild_set_percetange = 0.1
    # X_train, Y_train, X_valid, Y_valid = split_valid_set(X, Y, vaild_set_percetange)

    #Gussian distribution parameters
    train_data_size = X_train.shape[0]

    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((106,))
    mu2 = np.zeros((106,))
    for i in range(train_data_size):
        if Y_train[i] == 1:     # >50k
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((106, 106))
    sigma2 = np.zeros((106, 106))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [X_train[i] - mu1])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [X_train[i] - mu2])

    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2

    N1 = cnt1
    N2 = cnt2

    return mu1, mu2, shared_sigma, N1, N2


if __name__ == "__main__":
    trainData = pd.read_csv("data/train.csv")
    testData = pd.read_csv("data/test.csv")
    ans = pd.read_csv("data/correct_answer.csv")

#here is one more attribute in trainData
    x_train = dataProcess_X(trainData).drop(['native_country_ Holand-Netherlands'], axis=1).values
    x_test = dataProcess_X(testData).values
    y_train = dataProcess_Y(trainData).values
    y_ans = ans['label'].values

    vaild_set_percetange = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(x_train, y_train, vaild_set_percetange)
    mu1, mu2, shared_sigma, N1, N2 = train(X_train, Y_train)
    valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)

    mu1, mu2, shared_sigma, N1, N2 = train(x_train, y_train)
    sigma_inv = inv(shared_sigma)
    w = np.dot((mu1 - mu2), sigma_inv)
    X_t = x_test.T
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inv), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inv), mu2) + np.log(
        float(N1) / N2)
    a = np.dot(w, X_t) + b
    y = sigmoid(a)
    y_ = np.around(y).astype(np.int)
    df = pd.DataFrame({"id" : np.arange(1,16282), "label": y_})
    result = (np.squeeze(y_ans) == y_)
    print('Test acc = %f' % (float(result.sum()) / result.shape[0]))
    df = pd.DataFrame({"id": np.arange(1, 16282), "label": y_})
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df.to_csv(os.path.join(output_dir+'gd_output.csv'), sep='\t', index=False)










