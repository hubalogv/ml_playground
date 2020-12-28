import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, callbacks

random_seed = 2
from numpy.random import seed
seed(random_seed)
from tensorflow.random import set_seed
set_seed(random_seed)

# Read the data
train_data = pd.read_csv(r'C:\_ws\datasets\Housing Prices\train.csv', index_col='Id')
test_data = pd.read_csv(r'C:\_ws\datasets\Housing Prices\test.csv', index_col='Id')

def preprocess_data(data):
    """
    Do any preprocessing here which would be difficult to debug with the pipeline
    Args:
        data (pd.DataFrame):

    Returns (pd.DataFrame):

    """
    data['MSSubClass'] = data['MSSubClass'].apply(str)
    data['YearBuilt'] = 2011 - data['YearBuilt']
    # data.SalePrice = np.log1p(data.SalePrice)
    # data['YrSold'] = data['YrSold'].apply(str)
    # data['MoSold'] = data['MoSold'].apply(str)
    # data.drop(['YrSold'], axis=1)

    return data

# Remove rows with missing target, separate target from predictors
# train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)
train_data.drop(['SaleCondition'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
categoric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['object']]

all_cols = numeric_cols + categoric_cols

X = train_data.copy()
X_test = test_data.copy()

X = preprocess_data(X)
X_test = preprocess_data(X_test)

for col in numeric_cols:
    X[col] = X[col].fillna(0)
    X_test[col] = X_test[col].fillna(0)

for col in categoric_cols:

    X[col] = X[col].fillna('None')
    X_test[col] = X_test[col].fillna('None')

# print(X.isnull().values.any())
# print(X_test.isnull().values.any())

# label_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
# for col in categoric_cols:
#     print(col)
#     X[col] = label_encoder.fit_transform(X[col])
#     X_test[col] = label_encoder.transform(X_test[col])

X_test = pd.get_dummies(X_test ,drop_first=True)
X = pd.get_dummies(X, drop_first=True)


def build_model(input_dim):
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
    model.compile(loss='mae', optimizer='adam',  metrics=["accuracy"])
    return model

def build_model(input_dim):
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
    model.compile(loss='mae', optimizer='adam',  metrics=["accuracy"])
    return model

if 1:
    print(X.head())
    from sklearn.metrics import mean_absolute_error, mean_squared_log_error
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=0.8,
                                                          test_size=0.2,
                                                          random_state=random_seed)


    model = build_model(X_train.shape[1])

    checkpoint_filepath = 'hp_{val_loss:.4f}.h5'
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='max',
        save_best_only=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=50,
        epochs=300,
        callbacks=[model_checkpoint_callback]
    )

    # Show the learning curves
    history_df = pd.DataFrame(history.history)
    print(history_df.loc[:, ['loss', 'val_loss']])


    preds = model.predict(X_valid)

    print('MAE:', mean_absolute_error(y_valid, preds))
    print('MRSLE:', np.sqrt(mean_squared_log_error(y_valid, preds)))
    #
    # preds_test = model.predict(X_test)
    # # Save test predictions to file
    # output = pd.DataFrame({'Id': X_test.index,
    #                        'SalePrice': preds_test})
    # output.to_csv('submission.csv', index=False)
