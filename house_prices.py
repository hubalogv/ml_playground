import pandas as pd
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Sequential

def linear(input_dim):
    model = Sequential()
    model.add(layers.Dense(1, input_dim=input_dim))
    # Compile model
    model.compile(loss='mae', optimizer=optimizers.Adam(learning_rate=0.1))
    return model


def flat(input_dim):
    model = Sequential()
    model.add(layers.Dense(20, input_dim=input_dim))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(1))
    model.compile(loss='mae', optimizer=optimizers.Adam(learning_rate=0.1))
    return model

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

    data = data[['LotArea', 'GrLivArea', 'OverallQual', 'GarageCars']]
    return data

# Read the data
train_data = pd.read_csv(r'C:\_ws\datasets\Housing Prices\train.csv', index_col='Id')
test_data = pd.read_csv(r'C:\_ws\datasets\Housing Prices\test.csv', index_col='Id')

train_data = train_data.loc[train_data.GrLivArea < 4000]
# train_data = train_data.loc[train_data.SalePrice < 600000]

y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

# y = np.log1p(y)

X = train_data.copy()
X_test = test_data.copy()

X = preprocess_data(X)
X_test = preprocess_data(X_test)

numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
categoric_cols =  [cname for cname in X.columns if X[cname].dtype in ['object']]


numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean'))])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, categoric_cols)
    ])

# model = LinearRegression()
# model = RandomForestRegressor(n_estimators=100, random_state=4)
# model = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3)

constr = partial(flat, X.shape[1])
model = KerasRegressor(build_fn=constr, epochs=100, batch_size=50, verbose=1, )

my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

if 1:
    from sklearn.metrics import mean_absolute_error, mean_squared_log_error, max_error, median_absolute_error
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=0.8,
                                                          test_size=0.2,
                                                          random_state=0)

    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)

    # preds = np.exp(preds) - 1
    # y_valid = np.exp(y_valid) - 1

    print('MAX error:', max_error(y_valid, preds))
    print('Min error:', max(y_valid - preds))
    print('Median Absolutel error:', median_absolute_error(y_valid, preds))
    print('MAE:', mean_absolute_error(y_valid, preds))
    print('MRSLE:', np.sqrt(mean_squared_log_error(y_valid, preds)))

    preds_test = my_pipeline.predict(X_test)
    # Save test predictions to file
    output = pd.DataFrame({'Id': X_test.index,
                           'SalePrice': preds_test})
    output.to_csv('submission.csv', index=False)

if 1:
    from sklearn.model_selection import cross_val_score

    # Multiply by -1 since sklearn calculates *negative* MAE
    scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=5,
                                  scoring='neg_mean_absolute_error', verbose=True)

    print("Average MRSLE CV score:", scores.mean())
    # Average MRSLE CV score: 0.1402649418684216