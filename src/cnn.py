from tensorflow import keras
import tensorflow as tf
import numpy as np
from data import DataPreprocess
from data import label_to_letter
import cv2 as cv


class CNN:
    def __init__(self,load=False):
        self.learning_rate = 0.005
        self.epochs = 100
        self.weights_path = "trained_model/model1.ckpt"

        self.x_test = self.y_test = None
        self.x_train = self.y_train = None
        self.model = self.load_model() if load else self.get_model()

    def load_data(self):
        # get the data from the pre process file
        dp = DataPreprocess()
        self.x_train,self.y_train,self.x_test,self.y_test = dp.pre_process()

    def load_model(self):
        # load in saved model
        model = self.get_model()
        model.load_weights(self.weights_path)
        return model

    def get_model(self):
        # create the sequential model
        # model = keras.Sequential([
        #     keras.layers.Conv2D(50, (3, 3), input_shape=(28, 28, 1), activation='relu'),
        #     keras.layers.Conv2D(100, (3, 3), activation='relu'),
        #     keras.layers.MaxPool2D((2,2)),
        #     keras.layers.Dropout(0.25),
        #     keras.layers.Conv2D(250, (3, 3), activation='relu'),
        #     keras.layers.MaxPool2D((2, 2)),
        #     keras.layers.Dropout(0.25),
        #     keras.layers.Conv2D(400, (3, 3), activation='relu'),
        #     keras.layers.Conv2D(512, (3, 3), activation='relu'),
        #     keras.layers.Dropout(0.25),
        #     keras.layers.Flatten(input_shape=(28,28)),
        #     keras.layers.Dense(784, activation='relu'),
        #     keras.layers.Dense(392, activation='relu'),
        #     keras.layers.Dense(196, activation='relu'),
        #     keras.layers.Dense(98, activation='relu'),
        #     keras.layers.Dense(26, activation='softmax')
        # ])

        # model = keras.Sequential([
        #     keras.layers.Conv2D(50, (3, 3), input_shape=(28, 28, 1), activation='relu'),
        #     keras.layers.Conv2D(100, (3, 3), activation='relu'),
        #     keras.layers.MaxPool2D((2, 2)),
        #     keras.layers.Dropout(0.25),
        #     keras.layers.Conv2D(200, (3, 3), activation='relu'),
        #     keras.layers.MaxPool2D((2, 2)),
        #     keras.layers.Dropout(0.25),
        #     keras.layers.Conv2D(400, (3, 3), activation='relu'),
        #     keras.layers.MaxPool2D((2, 2)),
        #     keras.layers.Dropout(0.25),
        #     keras.layers.Flatten(input_shape=(28, 28)),
        #     keras.layers.Dense(784, activation='relu'),
        #     keras.layers.Dense(392, activation='relu'),
        #     keras.layers.Dense(196, activation='relu'),
        #     keras.layers.Dense(98, activation='relu'),
        #     keras.layers.Dense(26, activation='softmax')
        # ])

        model = keras.Sequential([
            keras.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(256, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Conv2D(512, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(784, activation='relu'),
            keras.layers.Dense(392, activation='relu'),
            keras.layers.Dense(196, activation='relu'),
            keras.layers.Dense(98, activation='relu'),
            keras.layers.Dense(26, activation='softmax')
        ])

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(lr=self.learning_rate),
            metrics=['accuracy']
        )

        return model

    def train(self):
        # save model
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.weights_path,save_weights_only=True,verbose=1)
        # train model on data
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, callbacks=[cp_callback])

    def test(self):
        # get testing accuracy
        accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)[1]
        print("accuracy: " + str(accuracy * 100) + "%")

    def predict(self, letter_img):
        # predicting a single image
        return np.argmax(self.model.predict(letter_img))


if __name__ == "__main__":
    def show_image(cnn,i):
        def imshow(x, y):
            def rotate(img):
                return cv.flip(cv.rotate(img, cv.ROTATE_90_CLOCKWISE), 1)
            img = rotate(x)
            cv.imshow(label_to_letter[y], img)
            cv.waitKey()
        letter_img = cnn.x_test[i]
        letter = cnn.y_test[i]
        print(label_to_letter[np.argmax(letter) + 1])
        letter_img_pred = letter_img.reshape(1, 28, 28, 1)
        letter_img_disp = letter_img.reshape(28, 28)
        letter_pred = cnn.predict(letter_img_pred)
        imshow(letter_img_disp, letter_pred + 1)

    cnn = CNN(load=False)
    cnn.load_data()
    cnn.train()
    print("DONE TRAINING")
    cnn.test()

    #show_image(cnn, 20000)
