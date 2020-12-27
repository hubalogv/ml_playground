import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from keras.models import Sequential
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier

train = pd.read_csv(r"C:\_ws\datasets\mnist\train.csv")
X_test = pd.read_csv(r"C:\_ws\datasets\mnist\test.csv")

Y_train = train["label"]

# Drop 'label' column
X_train = train.drop(labels=["label"], axis=1)

img_rows = 28
img_cols = 28

X_train = X_train.values.reshape(X_train.shape[0], img_rows, img_cols, 1)

input_shape = (img_rows, img_cols, 1)

random_seed = 2

X = X_train / 255.0
X_test = X_test / 255.0

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)
assert isinstance(Y_val, pd.Series)

def conv_model_0():
    mod = Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        # layers.Dropout(0.1),
        layers.BatchNormalization(),
        layers.Dense(128, activation="elu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dense(10, activation="softmax")])
    return mod

def cnn_model_2(input_shape, optimizer, loss):
    mod = Sequential()
    mod.add(layers.Conv2D(32, (3, 3), padding='Same', activation="relu", input_shape=input_shape))
    mod.add(layers.MaxPooling2D(pool_size=(2, 2)))
    mod.add(layers.Conv2D(32, (3, 3), activation="relu"))
    mod.add(layers.MaxPooling2D(pool_size=(2, 2)))
    mod.add(layers.Flatten())
    mod.add(layers.Dense(256, activation='relu'))
    mod.add(layers.Dense(10, activation='softmax'))
    mod.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    return mod


model = conv_model_0()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"]
)

history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=50, batch_size=200)

history_df = pd.DataFrame(history.history)
print(history_df.loc[:, ['loss', 'val_loss']])

test = X_test.values.reshape(-1, 28, 28, 1)

predictions = np.argmax(model.predict(test), axis=-1)

submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
    "Label": predictions})
submissions.to_csv("submission.csv", index=False, header=True)
