import numpy as np
import pandas as pd

def dataProcess(path_to_raw_data):
    rawData = pd.read_csv(path_to_raw_data)
    if 'label' in rawData.columns:
        x_train = str2flo(rawData.loc[:,'feature'])
        y_train = rawData.loc[:, 'label']
        return x_train, y_train
    x_test = str2flo(rawData.loc[:, 'feature'])
    return x_test

def str2flo(data):
    data = data.values.tolist()
    data = [x.split(" ") for x in data]
    data = [list(map(lambda x: float(x)/255, i)) for i in data]
    data = np.array(data)
    return data
