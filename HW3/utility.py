import pandas as pd
import numpy as np
from time import time
from matplotlib import pyplot as plt

def plot_training_history(r):
    # plot some data
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    # accuracies
    plt.plot(r.history['acc'], label='acc')
    plt.plot(r.history['val_acc'], label='val_acc')
    plt.legend()
    plt.show()


def plot_images(images, cls_true, cls_pred=None):
    name = ['angry','disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Get the i'th image and reshape the array.

        image = images[i].reshape(48, 48)

        # Ensure the noisy pixel-values are between 0 and 1.
        # image = np.clip(image, 0.0, 1.0)

        # Plot image.
        ax.imshow(image,
                  cmap='gray',
                  interpolation='nearest')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True:{0}".format(name[cls_true[i]])
        else:
            xlabel = "True:{0}, Pred:{1}".format(name[cls_true[i]], name[cls_pred[i]])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


class clean_data(object):
    _train = True

    def __init__(self, filename, train=True):
        self._train = train
        self._train_df = pd.read_csv(filename)
        self._train_df['feature'] = self._train_df['feature'].map(lambda x: np.array(list(map(float, x.split()))))
        self._image_size = self._train_df.feature[0].size
        self._image_shape = (int(np.sqrt(self._image_size)), int(np.sqrt(self._image_size)))
        self._dataNum = self._train_df.size
        self._feature = np.array(self._train_df.feature.map(lambda x: x.reshape(self._image_shape)).values.tolist())
        if self._train:
            self._label = self._train_df.label.values
            self._labelNum = self._train_df['label'].unique().size
            self._onehot = pd.get_dummies(self._train_df.label).values

    @property
    def distribution(self):
        return self._distribution

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_shape(self):
        return self._image_shape

    @property
    def dataNum(self):
        return self._dataNum

    @property
    def feature(self):
        return self._feature

    if _train:
        @property
        def label(self):
            return self._label

        @property
        def labelNum(self):
            return self._labelNum

        @property
        def onehot(self):
            return self._onehot
