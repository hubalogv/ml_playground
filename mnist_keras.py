import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
from keras import layers, callbacks
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt

random_seed = 2
from numpy.random import seed
seed(random_seed)
from tensorflow.random import set_seed
set_seed(random_seed)

def conv_model_0(): #
    mod = Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.BatchNormalization(),
        layers.Dense(128, activation="elu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")])
    mod.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )
    return mod

def conv_model_2():
    mod = Sequential()
    mod.add(layers.Conv2D(32, (3, 3), padding='Same', activation="relu", input_shape=(28, 28, 1)))
    mod.add(layers.MaxPooling2D(pool_size=(2, 2)))
    mod.add(layers.Conv2D(32, (3, 3), activation="relu"))
    mod.add(layers.MaxPooling2D(pool_size=(2, 2)))
    mod.add(layers.Flatten())
    mod.add(layers.Dense(256, activation='relu'))
    mod.add(layers.Dense(10, activation='softmax'))
    mod.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"]
    )
    return mod

def train():
    train = pd.read_csv(r"C:\_ws\datasets\mnist\train.csv")
    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)

    img_rows = 28
    img_cols = 28

    X_train = X_train.values.reshape(X_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    X = X_train / 255.0
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)
    print(X_train.shape)
    assert isinstance(Y_val, pd.Series)
    from matplotlib import pyplot as plt
    if 0:
        plt.imshow(X_train[0], cmap='gray')
        plt.show()

    model = conv_model_2()

    checkpoint_filepath = 'mnist.{epoch:02d}-{val_accuracy:.4f}.h5'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    history = model.fit(X_train, Y_train,
                        validation_data=(X_val, Y_val),
                        callbacks=[model_checkpoint_callback],
                        epochs=10, batch_size=200)

    history_df = pd.DataFrame(history.history)
    print(history_df.loc[:, ['loss', 'val_loss']])
    return model

def test(model):
    print(model.summary())

    X_test = pd.read_csv(r"C:\_ws\datasets\mnist\test.csv")
    X_test = X_test / 255.0
    test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)
    if 0:
        plt.imshow(test[0], cmap='gray')
        plt.show()

    pred = model.predict(test)
    # predictions = np.argmax(model.predict(test), axis=1)
    predictions = model.predict_classes(test, verbose=0)

    submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
        "Label": predictions})
    submissions.to_csv("submission.csv", index=False, header=True)

def eval(model):
    train = pd.read_csv(r"C:\_ws\datasets\mnist\train.csv")
    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)

    img_rows = 28
    img_cols = 28

    X_train = X_train.values.reshape(X_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    X = X_train / 255.0

    pred = np.argmax(model.predict(X), axis=1)

    print(list(pred))
    print(list(Y_train))
    print('accuracy: ', accuracy_score(Y_train, pred))



if __name__ == '__main__':
    # model = load_model('mnist.27-0.9910.h5')
    model = load_model('mnist.08-0.9826.h5')
    # model = train()
    # eval(model)
    test(model)