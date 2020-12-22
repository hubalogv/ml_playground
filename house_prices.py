import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Read the data
train_data = pd.read_csv(r'C:\_ws\datasets\Housing Prices\train.csv', index_col='Id')
test_data = pd.read_csv(r'C:\_ws\datasets\Housing Prices\test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
# train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
# categoric_cols = ['SaleCondition', 'SaleType', 'Functional', 'KitchenQual', 'Electrical', 'Street', 'Utilities', 'LotShape']
categoric_cols = ['SaleCondition', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2']

all_cols = numeric_cols + categoric_cols

X = train_data[all_cols].copy()
X_test = test_data[all_cols].copy()


# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean') # Your code here

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))]) # Your code here

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_cols),
        ('cat', categorical_transformer, categoric_cols)
    ])

my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=200, random_state=4))
])

if 1:
    from sklearn.metrics import mean_absolute_error, mean_squared_log_error
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          train_size=0.8,
                                                          test_size=0.2,
                                                          random_state=0)

    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)

    # Preprocessing of validation data, get predictions
    preds = my_pipeline.predict(X_valid)

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
                                  scoring='neg_mean_squared_log_error')

    print("Average MRSLE score:", np.sqrt(scores.mean()))