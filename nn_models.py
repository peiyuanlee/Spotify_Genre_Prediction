import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from keras.regularizers import L1L2, L2

class SequentialModel:
    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, config):
        """ Virtual Function """
        return

    def train(self, x, y, x_val, y_val, config):
        """ Virtual Function """
        return

    def evaluate(self, x_test, y_test, verbose=1):
        return self.model.evaluate(x_test,  y_test, verbose=verbose)

    def predict(self, x_test, verbose=1):
        return self.model.predict(x_test, verbose=verbose)

class Four_Layer_NN(SequentialModel):
    def __init__(self):
        super(Four_Layer_NN, self).__init__()

    def build_model(self, config):
        model = self.model

        input_shape = config["input_shape"]
        lr = config.get('lr', 0.001)
        decay = config.get("decay", 0.01)
        dropout = config.get('dropout', 0.2)

        model.add(Dense(64, kernel_regularizer=L2(l2=0.01), input_shape=input_shape, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(dropout))

        model.add(Dense(20, kernel_regularizer=L2(l2=0.01), activation='softmax'))
        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999,
                         amsgrad=False, epsilon=1e-8, decay=decay)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config):
        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                )
        return history

class Six_Layer_NN(SequentialModel):
    def __init__(self):
        super(Six_Layer_NN, self).__init__()

    def build_model(self, config):
        model = self.model

        input_shape = config["input_shape"]
        lr = config.get('lr', 0.001)
        decay = config.get("decay", 0.01)
        dropout = config.get('dropout', 0.2)

        model.add(Dense(32, kernel_regularizer=L2(l2=0.01), input_shape=input_shape, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(dropout))

        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(dropout))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(dropout))

        model.add(Dense(20, kernel_regularizer=L2(l2=0.01), activation='softmax'))
        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999,
                         amsgrad=False, epsilon=1e-8, decay=decay)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config):
        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                )
        return history

class Ten_Layer_NN(SequentialModel):
    def __init__(self):
        super(Ten_Layer_NN, self).__init__()

    def build_model(self, config):
        model = self.model

        input_shape = config["input_shape"]
        lr = config.get('lr', 0.001)
        decay = config.get("decay", 0.01)
        dropout = config.get('dropout', 0.2)

        model.add(Dense(32, kernel_regularizer=L2(l2=0.01), input_shape=input_shape, activation='elu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(dropout))

        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(dropout))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization(axis=1))

        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(dropout))

        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(dropout))

        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(dropout))

        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization(axis=1))
        model.add(Dropout(dropout))

        model.add(Dense(20, kernel_regularizer=L2(l2=0.01), activation='softmax'))
        optimizer = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999,
                         amsgrad=False, epsilon=1e-8, decay=decay)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        print(model.summary())
        print("Model compiled.")

    def train(self, x, y, x_val, y_val, config):
        history = self.model.fit(x, y, epochs=config['epochs'], batch_size=config['batch_size'],
                                 validation_data=(x_val, y_val), shuffle=True,
                                )
        return history