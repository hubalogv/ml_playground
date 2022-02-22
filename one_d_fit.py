import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, callbacks, optimizers, initializers
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, max_error, median_absolute_error

random_seed = 2
from numpy.random import seed
seed(random_seed)

X = np.random.rand(1000)
# y = np.sin(X * 10)
y = np.power(X, 5) + 1 + np.sin(X * 10)
# y = 4 * X + 3

# sns.scatterplot(x=X, y=y)
# plt.show()

if 1:
    from tensorflow.random import set_seed
    set_seed(random_seed)

    def linear(input_dim):
        model = Sequential()
        model.add(layers.Dense(4, input_dim=input_dim,
                               # kernel_initializer=initializers.random_normal(),
                               # bias_initializer=initializers.random_normal(),
                               ))
        model.add(layers.Activation('tanh'))
        model.add(layers.Dense(1,
                               # kernel_initializer=initializers.random_normal(),
                               # bias_initializer=initializers.random_normal(),
                               ))
        # Compile model
        model.compile(loss='mae', optimizer=optimizers.Adam(learning_rate=0.1))
        return model



    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=0.8,
                                                          test_size=0.2,
                                                          random_state=random_seed)


    model = linear(1)
    # model = linear(X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=100,
        epochs=300
    )
    print (model.layers[0].get_weights())
    print (model.layers[2].get_weights())
    weights0 = model.layers[0].get_weights()
    weights2 = model.layers[2].get_weights()
    w_array = weights0[0]
    b_array = weights0[1]

    for i in range(len(w_array[0])):
        w = w_array[0][i]
        b = b_array[i]
        data = X * w + b
        sns.scatterplot(x=X, y=data)
    plt.show()

    for i in range(len(w_array[0])):
        w = w_array[0][i]
        b = b_array[i]
        data = X * w + b
        sns.scatterplot(x=X, y=np.tanh(data)*weights2[0][i][0])
    plt.show()

    if 0:
        # Show the learning curves
        history_df = pd.DataFrame(history.history)
        history_df.loc[:, ['loss', 'val_loss']].plot()
        plt.show()

    if 1:
        preds = model.predict(X_valid)[:,0]
        sns.scatterplot(x=X, y=y)
        sns.scatterplot(x=X_valid, y=preds)

        print('MAX error:', max_error(y_valid, preds))
        print('Min error:', max(y_valid - preds))
        print('Median Absolutel error:', median_absolute_error(y_valid, preds))
        print('MAE:', mean_absolute_error(y_valid, preds))

        # sns.displot(y_valid - preds)

        plt.show()



