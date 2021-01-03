import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, callbacks, regularizers, optimizers
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, max_error, median_absolute_error
from sklearn.model_selection import KFold

random_seed = 2
from numpy.random import seed
seed(random_seed)
from tensorflow.random import set_seed
set_seed(random_seed)

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
        self.x['MSSubClass'] = self.x['MSSubClass'].apply(str)
        self.x['YearBuilt'] = 2011 - self.x['YearBuilt']
        # data.SalePrice = np.log1p(data.SalePrice)
        # data['YrSold'] = data['YrSold'].apply(str)
        # data['MoSold'] = data['MoSold'].apply(str)
        # data.drop(['YrSold'], axis=1)

        numeric_cols = [cname for cname in self.x.columns if self.x[cname].dtype in ['int64', 'float64']]
        categoric_cols = [cname for cname in self.x.columns if self.x[cname].dtype in ['object']]
        all_cols = numeric_cols + categoric_cols

        for col in numeric_cols:
            self.x[col] = self.x[col].fillna(0)
            self.x_test[col] = self.x_test[col].fillna(0)

        for col in categoric_cols:
            self.x[col] = self.x[col].fillna('None')
            self.x_test[col] = self.x_test[col].fillna('None')

        # print(X.isnull().values.any())
        # print(X_test.isnull().values.any())

        # label_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        # for col in categoric_cols:
        #     print(col)
        #     X[col] = label_encoder.fit_transform(X[col])
        #     X_test[col] = label_encoder.transform(X_test[col])

        self.x_test = pd.get_dummies(self.x_test, drop_first=True)
        self.x = pd.get_dummies(self.x, drop_first=True)

        self.x, self.x_test = self.x.align(self.x_test, join='left', axis=1)

        self.x_test = self.x_test.fillna(0)

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
            model.add(layers.Dense(400, input_dim=input_dim))
            model.add(layers.Dropout(0.3))
            # model.add(layers.BatchNormalization())
            model.add(layers.Activation('relu'))
            # model.add(layers.Dense(400, activation='relu'))
            model.add(layers.Dense(1, kernel_initializer='normal'))
            # Compile model
            lr_schedule = optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.01,
                decay_steps=10000,
                decay_rate=0.8)
            model.compile(loss='mae', optimizer=optimizers.Adam(learning_rate=lr_schedule),  metrics=["mean_squared_logarithmic_error"])
            return model

        elif model_id == 'lin':
            model = Sequential()
            model.add(layers.Dense(1, input_dim=input_dim))
            # Compile model
            model.compile(loss='mae', optimizer='adam',  metrics=["mean_squared_logarithmic_error"])
            return model
        else:
            raise ValueError('unknown model id')

    def train_eval(self, x_train, x_valid, y_train, y_valid):
        # x_train, x_valid, y_train, y_valid = train_test_split(self.x, self.y,
        #                                                       train_size=0.8,
        #                                                       test_size=0.2,
        #                                                       random_state=random_seed)


        model = self.build_model('flat')

        checkpoint_filepath = 'hp_{loss:.4f}.h5'
        model_checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor='loss',
            mode='max',
            save_best_only=True)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_valid, y_valid) if x_valid else None,
            batch_size=50,
            epochs=2000,
            verbose=0,
            callbacks=[model_checkpoint_callback]
        )

        model.save('house_prices.h5')

        # Show the learning curves
        history_df = pd.DataFrame(history.history)
        if x_valid:
            history_df.loc[:, ['loss', 'val_loss']].plot()
        else:
            history_df.loc[:, ['loss']].plot()
            print('loss: ', history_df.loc[-1:, ['loss']])

        if x_valid:

            preds = model.predict(x_valid)[:,0]
            # plt.show()

            results = {}
            results['MAX error'] = max_error(y_valid, preds)
            results['Min error:'] = max(y_valid - preds)
            results['Median Absolutel error'] = median_absolute_error(y_valid, preds)
            results['MAE']= mean_absolute_error(y_valid, preds)
            results['MRSLE'] = np.sqrt(mean_squared_log_error(y_valid, preds))
            print(results)

        # sns.displot(y_valid - preds)
        # plt.show()
        return model

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
        metrics = []
        for train_index, test_index in skf.split(self.x, self.y):
            x_train, x_valid = self.x.iloc[train_index], self.x.iloc[test_index]
            y_train, y_valid = self.y.iloc[train_index], self.y.iloc[test_index]
            metrics.append(self.train_eval(x_train, x_valid, y_train, y_valid)[1])

        df = pd.DataFrame(metrics)
        print('Agv MRLSE: ', df['MRSLE'].mean())
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

    if 0:
        pl.cross_validation()
    if 0:
        model = pl.train_eval(pl.x, None, pl.y, None)
        pl.test(model)