from tensorflow import keras
from data import DataPreprocess


class CNN:
    def __init__(self):
        self.learning_rate = 0.01
        self.epochs = 50

        self.x_test = self.y_test = None
        self.x_train = self.y_train = None
        self.model = self.get_model()

    def load_data(self):
        dp = DataPreprocess()
        self.x_train,self.y_train,self.x_test,self.y_test = dp.pre_process()

        self.y_train = self.y_train.reshape(self.y_train.shape[0],1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0],1)

        self.y_train = self.y_train-1
        self.y_test = self.y_test-1

        self.y_train = keras.utils.to_categorical(self.y_train, 26)
        self.y_test = keras.utils.to_categorical(self.y_test, 26)

    def get_model(self):
        model = keras.Sequential([
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
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs)

    def test(self):
        accuracy = self.model.evaluate(self.x_test, self.y_test)[1]
        print("accuracy: %" + str(accuracy * 100))


if __name__ == "__main__":
    cnn = CNN()
    cnn.load_data()
    cnn.train()
    print("DONE TRAINING")
    cnn.test()
