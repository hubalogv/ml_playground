import pandas as pd
import numpy as np
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, callbacks, regularizers, optimizers, losses
from tensorflow.keras import backend as K
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, max_error, median_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

random_seed = 2
from numpy.random import seed
seed(random_seed)
from tensorflow.random import set_seed
set_seed(random_seed)

class CustomCallback(callbacks.Callback):


    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        # print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        current_decayed_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        if divmod(epoch, 100)[1] == 0:
            print("epoch: {}, loss: {:0.7f}, val_loss: {:0.7f}, lr: {:0.7f}".format(epoch,
                                                                          logs['loss'],
                                                                          logs['val_loss'] if 'val_loss' in logs.keys() else 0,
                                                                          current_decayed_lr))
def rmsle(y_true, y_pred):
  return tf.math.sqrt(tf.reduce_mean(losses.mean_squared_logarithmic_error(y_true, y_pred)))

# def rmsle(y_true, y_pred):
#    return np.sqrt(mean_squared_log_error(y_true, y_pred))

class HousePricesPipeline(object):

    def __init__(self):
        super(HousePricesPipeline, self).__init__()
        self.x = None  # type: pd.DataFrame
        self.y = None  # type: pd.DataFrame
        self.x_test = None  # type: pd.DataFrame
        self.y_test = None  # type: pd.DataFrame

    def load_data(self):
        train_data = pd.read_csv(r'C:\_ws\datasets\Housing Prices\train.csv', index_col='Id')
        test_data = pd.read_csv(r'C:\_ws\datasets\Housing Prices\test.csv', index_col='Id')
        train_data = train_data.loc[train_data.GrLivArea < 4000]
        self.y = train_data.SalePrice
        train_data.drop(['SalePrice'], axis=1, inplace=True)

        self.x = train_data
        self.x_test = test_data

    def pre_process_data(self):
        # data.SalePrice = np.log1p(data.SalePrice)
        # data['YrSold'] = data['YrSold'].apply(str)
        # data['MoSold'] = data['MoSold'].apply(str)
        # data.drop(['YrSold'], axis=1)

        x_length = self.x.shape[0]

        all_data = self.x.append(self.x_test)  # type: pd.DataFrame
        all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
        # all_data['MoSold'] = all_data['MoSold'].apply(str)
        all_data['YearBuilt'] = 2011 - all_data['YearBuilt']
        # all_data['YearRemodAdd'] = 2011 - all_data['YearRemodAdd']


        # garage items are fine, default fillers should work fine

        fill_avg_num_cols = []
        fill_zero_num_cols = ['Frontage', 'MasVnrArea',
                              'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtHalfBath', 'BsmtFullBath',
                              ]

        fill_na_str_cols = ['MasVnrType']
        fill_avg_str_cols = ['Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType',
                             'MSZoning', 'Utilities', 'Functional',
                             'BsmtFinType1', 'BsmtFinType2']

        all_data['GarageCars'] = all_data['GarageCars'].fillna(1)

        for col in fill_avg_str_cols:
            all_data[col] = all_data[col].fillna(self.x[col].value_counts().idxmax())


        numeric_cols = [cname for cname in self.x.columns if self.x[cname].dtype in ['int64', 'float64']]
        categoric_cols = [cname for cname in self.x.columns if self.x[cname].dtype in ['object']]
        all_cols = numeric_cols + categoric_cols

        for col in numeric_cols:
            all_data[col] = all_data[col].fillna(0)

        for col in categoric_cols:
            all_data[col] = all_data[col].fillna('None')

        # print(X.isnull().values.any())
        # print(X_test.isnull().values.any())

        # label_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        # for col in categoric_cols:
        #     print(col)
        #     X[col] = label_encoder.fit_transform(X[col])
        #     X_test[col] = label_encoder.transform(X_test[col])

        all_data = pd.get_dummies(all_data, drop_first=True)

        self.x = all_data.iloc[:x_length]
        self.x_test = all_data.iloc[x_length:]

        if 0:
            scaler = StandardScaler()
            self.x[self.x.columns] = scaler.fit_transform(self.x[self.x.columns])
            self.x_test[self.x.columns] = scaler.transform(self.x_test[self.x_test.columns])

        if 0:
            print (self.x.shape[0])
            print (self.x_test.shape[0])
            print(self.x.isna().values.any())
            print(self.x_test.isna().values.any())

    def build_model(self, model_id=None):
        input_dim = self.x.shape[1]
        if model_id == 'deep':
            model = Sequential()
            model.add(layers.BatchNormalization(input_dim=input_dim))
            model.add(layers.Dense(800, kernel_initializer='normal', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(1600, kernel_initializer='normal', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(400, kernel_initializer='normal', activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.3))
            model.add(layers.Dense(400, kernel_initializer='normal', activation='relu'))
            # model.add(Dense(25, kernel_initializer='normal', activation='relu'))
            model.add(layers.Dense(1, kernel_initializer='normal'))
            # Compile model
            model.compile(loss='mae', optimizer='adam',  metrics=["mean_squared_logarithmic_error"])
            return model

        elif model_id == 'flat':
            model = Sequential()
            model.add(layers.Dense(400, input_dim=input_dim,
                                   kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                    bias_regularizer=regularizers.l2(1e-4),
                                    activity_regularizer=regularizers.l2(1e-5)))
            model.add(layers.Dropout(0.3))
            # model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            # model.add(layers.Dense(400, activation='relu'))
            model.add(layers.Dense(1, kernel_initializer='normal'))
            # Compile model
            lr_schedule = optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.05,
                decay_steps=10000,
                decay_rate=0.8)
            model.compile(loss='mae', optimizer=optimizers.Adam(learning_rate=lr_schedule),  metrics=[rmsle])
            return model

        elif model_id == 'lin':
            model = Sequential()
            model.add(layers.Dense(1, input_dim=input_dim))
            # Compile model
            model.compile(loss=rmsle, optimizer='adam',  metrics=["mae"])
            return model
        else:
            raise ValueError('unknown model id')

    def train_eval(self, x_train, x_valid, y_train, y_valid):
        # x_train, x_valid, y_train, y_valid = train_test_split(self.x, self.y,
        #                                                       train_size=0.8,
        #                                                       test_size=0.2,
        #                                                       random_state=random_seed)

        is_validated = isinstance(x_valid, pd.DataFrame)

        model = self.build_model('flat')

        checkpoint_filepath = 'hp_{loss:.4f}.h5'
        model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='loss',
            mode='max',
            save_best_only=True)

        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss' if is_validated else 'loss',
            patience=500,
            min_delta=500 if is_validated else 50,
            restore_best_weights=True)

        log_dir = r'C:\_ws\ML\logs\fit' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_valid, y_valid) if is_validated else None,
            batch_size=50,
            epochs=2001,
            verbose=0,
            callbacks=[tensorboard_callback, es_callback, CustomCallback()]
        )

        model.save('house_prices.h5')

        # Show the learning curves
        history_df = pd.DataFrame(history.history)
        if is_validated:
            history_df.loc[:, ['loss', 'val_loss']].plot()
            # plt.show()
        else:
            history_df.loc[:, ['loss']].plot()
            # plt.show()
            # print('loss: ', history_df.loc[:, ['loss']])

        if is_validated:

            preds = model.predict(x_valid)[:,0]
            # print(preds)
            # plt.show()

            results = {}
            results['MAX error'] = max_error(y_valid, preds)
            results['Min error:'] = max(y_valid - preds)
            results['Median Absolutel error'] = median_absolute_error(y_valid, preds)
            results['MAE']= mean_absolute_error(y_valid, preds)
            results['RMSLE'] = np.sqrt(mean_squared_log_error(y_valid, preds))
            print(results)
        else:
            results = {}

        # sns.displot(y_valid - preds)
        # plt.show()
        return model, results

    def test(self, model):
            preds_test = model.predict(self.x_test)[:,0]
            # Save test predictions to file
            output = pd.DataFrame({'Id': self.x_test.index,
                                   'SalePrice': preds_test})
            output.to_csv('submission.csv', index=False)

    def cross_validation(self):
        n_folds = 5
        skf = KFold(n_splits=n_folds)
        # skf.get_n_splits(self.x, self.y)
        preds = []
        metrics = []
        for train_index, test_index in skf.split(self.x, self.y):
            x_train, x_valid = self.x.iloc[train_index], self.x.iloc[test_index]
            y_train, y_valid = self.y.iloc[train_index], self.y.iloc[test_index]
            model, met = self.train_eval(x_train, x_valid, y_train, y_valid)
            metrics.append(met)
            preds.append(model.predict(self.x_test))

        summed = np.average(np.array(preds), axis=0)
        print( summed)
        output = pd.DataFrame({'Id': self.x_test.index,
                               'SalePrice': summed[:,0]})
        output.to_csv('submission.csv', index=False)

        df = pd.DataFrame(metrics)
        print('Agv RMLSE: ', df['RMSLE'].mean())
        return metrics


if __name__ == '__main__':
    pl = HousePricesPipeline()
    pl.load_data()
    pl.pre_process_data()
    if 0:
        x_train, x_valid, y_train, y_valid = train_test_split(pl.x, pl.y,
                                                          train_size=0.8,
                                                          test_size=0.2,
                                                          random_state=random_seed)
        pl.train_eval(x_train, x_valid, y_train, y_valid)

    if 1:
        pl.cross_validation()
    if 0:
        model = pl.train_eval(pl.x, None, pl.y, None)[0]
        pl.test(model)