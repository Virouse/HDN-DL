# coding: utf-8
import os.path
import gzip
import pickle
import os
import numpy as np
# from sklearn.model_selection import train_test_split

url_base = 'http://yann.lecun.com/exdb/mnist/'  # mnist官网，下载失败可以从此处下载，文件名见下方字典
key_file = {  # 字典存储下载好的文件名
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

dataset_dir = os.path.dirname(
    os.path.abspath(__file__)) + "/MNIST_data"  # os.path.abspath(__file__)返回此片代码绝对路径，os.path.dirname（）返回此路径文件的目录
save_file = dataset_dir + "/mnist.pkl"

train_num = 60000  # mnist数据集共60000个数据用于训练，10000个数据用于测试
test_num = 10000
img_size = 784


def _load_img(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)  # 共image_siza=28*28列，-1为自适应行数
    print("Done")

    return data


def _load_label(file_name):
    file_path = dataset_dir + "/" + file_name

    print("Converting " + file_name + " to NumPy Array ...")
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _convert_numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


def init_mnist():
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集

    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 784)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

# (X_train, y_train), (X_test, y_test) = load_mnist(flatten=False, normalize=False)
# X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.1)
# X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.2)
# X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, test_size=0.25)
# # print(len(y_train), y_test.shape)
# print(X_train.shape)
# print(np.unique(y_test))