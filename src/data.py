import cv2 as cv
import numpy as np
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
        cv.imshow(label_to_letter[y[i]], img)
        cv.waitKey()

    def rotate(self,img):
        return cv.flip(cv.rotate(img, cv.ROTATE_90_CLOCKWISE), 1)

    def pre_process(self):
        def normalize(image):
            return (image - np.mean(image)) / (np.std(image))

        self.x_train, self.x_test = normalize(self.x_train), normalize(self.x_test)
        return self.x_train,self.y_train,self.x_test,self.y_test

    def get_data(self):
        return self.x_train,self.y_train,self.x_test,self.y_test


if __name__ == "__main__":
    dp = DataPreprocess()
    x_train,y_train,x_test,y_test = dp.get_data()
    print(x_train.shape)
    dp.imshow(87569,x_train,y_train)