import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

train = pd.read_csv(r"C:\_ws\datasets\mnist\train.csv")
test = pd.read_csv(r"C:\_ws\datasets\mnist\test.csv")

Y = train["label"]

# Drop 'label' column
X = train.drop(labels=["label"], axis=1)

random_seed = 2

X = X/255.0
X_test = test/255.0

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=random_seed)
assert isinstance(Y_val, pd.Series)

def dense_model_0():
    model = Sequential()
    model.add(layers.Dense(600, input_dim=784, activation='relu'))
    model.add(layers.Dropout(0.3),)
    model.add(layers.Dense(200, activation='relu'))
    model.add(layers.Dropout(0.3),)
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=dense_model_0, epochs=50, batch_size=200)
# model = LogisticRegression(max_iter=50)
# model = RidgeClassifier()

model.fit(X_train, Y_train)

predictions = model.predict(X_val)

print('score: ', accuracy_score(Y_val, predictions))

if 0:
    model.fit(X, Y)
    results = model.predict(X_test)

    output = pd.DataFrame({'ImageId': range(1,len(results)+1),
                           'Label': results})
    output.to_csv('submission.csv', index=False)
