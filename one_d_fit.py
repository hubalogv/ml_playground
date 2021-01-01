import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

random_seed = 2
from numpy.random import seed
seed(random_seed)


X = np.random.rand(1000)
# y = np.sin(X * 10)
y = 4 * X + 3

# sns.scatterplot(X, y)
# plt.show()

if 1:
    from tensorflow.random import set_seed
    set_seed(random_seed)

    def linear(input_dim):
        model = Sequential()
        model.add(layers.Dense(1, input_dim=input_dim))
        # Compile model
        model.compile(loss='mae', optimizer='adam')
        return model

    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers, callbacks
    from sklearn.metrics import mean_absolute_error, mean_squared_log_error, max_error, median_absolute_error


    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=0.8,
                                                          test_size=0.2,
                                                          random_state=random_seed)


    model = linear(1)
    # model = linear(X_train.shape[1])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=50,
        epochs=1000,
    )

    # Show the learning curves
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot()

    print (model.layers[0].get_weights())

    preds = model.predict(X_valid)[:,0]
    # plt.show()

    print('MAX error:', max_error(y_valid, preds))
    print('Min error:', max(y_valid - preds))
    print('Median Absolutel error:', median_absolute_error(y_valid, preds))
    print('MAE:', mean_absolute_error(y_valid, preds))

    # sns.displot(y_valid - preds)
    # plt.show()


