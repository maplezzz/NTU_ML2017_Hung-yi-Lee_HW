import numpy as np
import pandas as pd


def dataProcess(path_to_raw_data):
    rawData = pd.read_csv(path_to_raw_data)
    if 'label' in rawData.columns:
        x_train = str2flo(rawData.loc[:, 'feature'])
        x_train_image = x_train.reshape(x_train.shape[0], 48, 48, 1)
        temp = rawData.loc[:, 'label'].values
        n_train = np.max(temp)+1
        y_train = np.eye(n_train)[temp]
        return x_train, x_train_image, y_train
    x_test = str2flo(rawData.loc[:, 'feature'])
    x_test_image = x_test.reshape(x_test.shape[0], 48, 48, 1)
    return x_test, x_test_image


def str2flo(data):
    data = data.values.tolist()
    data = [x.split(" ") for x in data]
    data = [list(map(lambda x: float(x)/255, i)) for i in data]
    data = np.array(data)
    return data


if __name__ == "__main__":
    x_train, x_train_image, y_train = dataProcess("data/train.csv")
    x_test, x_test_image = dataProcess("data/test.csv")
    np.savez("data/Train.npz", Label = y_train, Image = x_train_image)
    np.savez("data/Test.npz", Image = x_test_image)

