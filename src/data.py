import cv2 as cv
import numpy as np
from tensorflow import keras
import idx2numpy

label_to_letter = {
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e',
    6: 'f',
    7: 'g',
    8: 'h',
    9: 'i',
    10: 'j',
    11: 'k',
    12: 'l',
    13: 'm',
    14: 'n',
    15: 'o',
    16: 'p',
    17: 'q',
    18: 'r',
    19: 's',
    20: 't',
    21: 'u',
    22: 'v',
    23: 'w',
    24: 'x',
    25: 'y',
    26: 'z',
}


class DataPreprocess:
    def __init__(self):
        self.train_file_labels = '../data/emnist-letters-train-labels-idx1-ubyte'
        self.train_file_imgs = '../data/emnist-letters-train-images-idx3-ubyte'
        self.test_file_labels = '../data/emnist-letters-test-labels-idx1-ubyte'
        self.test_file_imgs = '../data/emnist-letters-test-images-idx3-ubyte'

        self.x_train = idx2numpy.convert_from_file(self.train_file_imgs)
        self.y_train = idx2numpy.convert_from_file(self.train_file_labels)
        self.x_test = idx2numpy.convert_from_file(self.test_file_imgs)
        self.y_test = idx2numpy.convert_from_file(self.test_file_labels)

    def imshow(self,i,x,y):
        img = self.rotate(x[i])
        img = x[i]
        cv.imshow(label_to_letter[y[i]], img)
        cv.waitKey()

    # rotate the image clockwise and flip over y axis
    def rotate(self,img):
        return cv.flip(cv.rotate(img, cv.ROTATE_90_CLOCKWISE), 1)

    # pre process the image
    def pre_process(self):
        # normalize [x-mean(x)]/std(x)
        def normalize(image):
            return (image - np.mean(image)) / (np.std(image))

        # threshold the image
        self.x_train = np.uint8((self.x_train > 0) * 255)
        self.x_test = np.uint8((self.x_test > 0) * 255)

        # call normalize function
        self.x_train, self.x_test = normalize(self.x_train), normalize(self.x_test)

        # reshape for network
        self.x_train = self.x_train.reshape(self.x_train.shape[0],28,28,1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0],28,28,1)

        # reshape for network
        self.y_train = self.y_train.reshape(self.y_train.shape[0],1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0],1)

        # fix labels
        self.y_train = self.y_train-1
        self.y_test = self.y_test-1

        # get categorical outputs
        self.y_train = keras.utils.to_categorical(self.y_train, 26)
        self.y_test = keras.utils.to_categorical(self.y_test, 26)
        return self.x_train,self.y_train,self.x_test,self.y_test

    def get_data(self):
        return np.uint8((self.x_train > 0) * 255),self.y_train,self.x_test,self.y_test


if __name__ == "__main__":
    dp = DataPreprocess()
    x_train,y_train,x_test,y_test = dp.get_data()
    i = np.where(y_train==5)
    print(i[0][0])
    dp.imshow(i[0][0],x_train,y_train)