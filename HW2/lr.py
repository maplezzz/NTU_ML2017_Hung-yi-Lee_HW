import pandas as pd
import numpy as np
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



def valid(X, Y, w):
    a = np.dot(w,X.T)
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y) == y_)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return y_

def train(X_train, Y_train):
    # valid_set_percentage = 0.1
    # X_train, Y_train, X_valid, Y_valid = split_valid_set(X, Y, valid_set_percentage)

    w = np.zeros(len(X_train[0]))

    l_rate = 0.001
    batch_size = 32
    train_dataz_size = len(X_train)
    step_num = int(floor(train_dataz_size / batch_size))
    epoch_num = 300
    list_cost = []

    total_loss = 0.0
    for epoch in range(1, epoch_num):
        total_loss = 0.0
        X_train, Y_train = _shuffle(X_train, Y_train)

        for idx in range(1, step_num):
            X = X_train[idx*batch_size:(idx+1)*batch_size]
            Y = Y_train[idx*batch_size:(idx+1)*batch_size]

            s_grad = np.zeros(len(X[0]))


            z = np.dot(X, w)
            y = sigmoid(z)
            loss = y - np.squeeze(Y)

            cross_entropy = -1 * (np.dot(np.squeeze(Y.T), np.log(y)) + np.dot((1 - np.squeeze(Y.T)), np.log(1 - y)))/ len(Y)
            total_loss += cross_entropy

            grad = np.sum(-1 * X * (np.squeeze(Y) - y).reshape((batch_size, 1)), axis=0)
            # grad = np.dot(X.T, loss)
            w = w - l_rate * grad

            # s_grad += grad ** 2
            # ada = np.sqrt(s_grad)
            # w = w - l_rate * grad / ada

        list_cost.append(total_loss)

    # valid(X_valid, Y_valid, w)
    plt.plot(np.arange(len(list_cost)), list_cost)
    plt.title("Train Process")
    plt.xlabel("epoch_num")
    plt.ylabel("Cost Function (Cross Entropy)")
    plt.savefig(os.path.join(os.path.dirname(output_dir), "TrainProcess"))
    plt.show()

    return w

if __name__ == "__main__":
    trainData = pd.read_csv("data/train.csv")
    testData = pd.read_csv("data/test.csv")
    ans = pd.read_csv("data/correct_answer.csv")

    # here is one more attribute in trainData
    x_train = dataProcess_X(trainData).drop(['native_country_ Holand-Netherlands'], axis=1).values
    x_test = dataProcess_X(testData).values
    y_train = dataProcess_Y(trainData).values
    y_ans = ans['label'].values


    x_test = np.concatenate((np.ones((x_test.shape[0], 1)), x_test), axis=1)
    x_train = np.concatenate((np.ones((x_train.shape[0], 1)),x_train), axis=1)

    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(x_train, y_train, valid_set_percentage)

    w_train = train(X_train, Y_train)
    valid(X_train, Y_train, w_train)

    w = train(x_train, y_train)

    y_ = valid(x_test, y_ans, w)

    df = pd.DataFrame({"id": np.arange(1, 16282), "label": y_})
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df.to_csv(os.path.join(output_dir + 'lr_output.csv'), sep='\t', index=False)

