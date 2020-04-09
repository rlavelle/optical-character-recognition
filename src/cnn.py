from tensorflow import keras
import tensorflow as tf
from data import DataPreprocess


class CNN:
<<<<<<< HEAD
    def __init__(self,load=False):
=======
    def __init__(self):
>>>>>>> 36cba9cb5654d232d1043c60f153f1d4feb05cb5
        self.learning_rate = 0.005
        self.epochs = 100
        self.weights_path = "training_model/cp.ckpt"

        self.x_test = self.y_test = None
        self.x_train = self.y_train = None
        self.model = self.load_model() if load else self.get_model()

    def load_data(self):
        dp = DataPreprocess()
        self.x_train,self.y_train,self.x_test,self.y_test = dp.pre_process()

        self.x_train = self.x_train.reshape(self.x_train.shape[0],28,28,1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0],28,28,1)

        self.y_train = self.y_train.reshape(self.y_train.shape[0],1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0],1)

        self.y_train = self.y_train-1
        self.y_test = self.y_test-1

        self.y_train = keras.utils.to_categorical(self.y_train, 26)
        self.y_test = keras.utils.to_categorical(self.y_test, 26)

    def load_model(self):
        model = self.get_model()
        model.load_weights(self.weights_path)
        return model

    def get_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(14, (3, 3), input_shape=(28, 28, 1), activation='relu'),
<<<<<<< HEAD
            keras.layers.Conv2D(28, (3, 3), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            keras.layers.Conv2D(28, (3, 3), activation='relu'),
=======
            keras.layers.Conv2D(28, (5, 5), activation='relu'),
>>>>>>> 36cba9cb5654d232d1043c60f153f1d4feb05cb5
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            keras.layers.Conv2D(28, (7, 7), activation='relu'),
            keras.layers.MaxPool2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(input_shape=(28,28)),
            keras.layers.Dense(784, activation='relu'),
            keras.layers.Dense(392, activation='relu'),
            keras.layers.Dense(392, activation='relu'),
            keras.layers.Dense(196, activation='relu'),
            keras.layers.Dense(196, activation='relu'),
            keras.layers.Dense(98, activation='relu'),
            keras.layers.Dense(49, activation='relu'),
            keras.layers.Dense(26, activation='softmax')
        ])

        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(lr=self.learning_rate),
            metrics=['accuracy']
        )

        return model

    def train(self):
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.weights_path,ave_weights_only=True)
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, callbacks=[cp_callback])

    def test(self):
        accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=False)[1]
        print("accuracy: %" + str(accuracy * 100))


if __name__ == "__main__":
    cnn = CNN(load=True)
    cnn.load_data()
    cnn.train()
    print("DONE TRAINING")
    cnn.test()
