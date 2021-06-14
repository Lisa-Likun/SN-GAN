import scipy.misc
import math
import numpy as np
import os
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras import backend as K

# conv out size
def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

# load data from datasets
def get_image(batch_file, is_grayscale=False):
    if is_grayscale:
        return scipy.misc.imread(batch_file, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(batch_file).astype(np.float)
        
def save_images(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def load_cifar10():
    # data_dir = "D:\test11111111111111/test11111111111111/cifar-10-batches-py"
    #
    # #filenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in xrange(1, 6)]
    # filenames = ['data_batch_%d.bin' % i for i in xrange(1, 6)]
    # filenames.append(os.path.join(data_dir, 'test_batch'))
    filename = 'cifar-10-batches-py'
    # origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    path = 'D:/test11111111111111/test11111111111111/cifar-10-batches-py'
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
     #############################################
    #
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # fpath = 'D:/test11111111111111/test11111111111111/cifar-10-batches-py/data_batch_1'
    # (x_train[0* 10000:1 * 10000, :, :, :],y_train[0 * 10000:1 * 10000]) = load_batch(fpath)
    # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    ####################################
    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
    dataX = x_train
    labely = y_train
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(dataX)
    np.random.seed(seed)
    np.random.shuffle(labely)

    y_vec = np.zeros((len(labely), 10), dtype=np.float)
    for i, label in enumerate(labely):
        y_vec[i, labely[i]] = 1.0

    return dataX / 255., y_vec
