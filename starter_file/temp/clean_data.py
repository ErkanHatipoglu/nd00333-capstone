from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import os
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

global X_train, X_test, y_train, y_test

def clean_data(data):

    # Copy data
    print("0")
    X = data.to_pandas_dataframe()
    X.set_index('Id',inplace=True)
    print(X.head())
    print()

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice 

    # Remove target and 'Utilities' 
    X.drop(['SalePrice', 'Utilities'], axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    # Select object columns
    categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]

    # Select numeric columns
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64','float64']]

    # Imputation lists

    # imputation to null values of these numerical columns need to be 'constant'
    constant_num_cols = ['GarageYrBlt', 'MasVnrArea']
    #constant_num_cols = ['MasVnrArea']
    print("constant_num_cols")
    print(constant_num_cols)
    print

    # imputation to null values of these numerical columns need to be 'mean'
    mean_num_cols = list(set(numerical_cols).difference(set(constant_num_cols)))
    print("mean_num_cols")
    print(mean_num_cols)
    print()

    # imputation to null values of these categorical columns need to be 'constant'
    constant_categorical_cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
    print("constant_categorical_cols")
    print(constant_categorical_cols)
    print()

    # imputation to null values of these categorical columns need to be 'most_frequent'
    mf_categorical_cols = list(set(categorical_cols).difference(set(constant_categorical_cols)))
    print("mf_categorical_cols")
    print(mf_categorical_cols)
    print()

    my_cols = constant_num_cols + mean_num_cols + constant_categorical_cols + mf_categorical_cols
    print("my_cols")
    print(my_cols)
    print()

    # Define transformers
    # Preprocessing for numerical data

    numerical_transformer_m = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])

    numerical_transformer_c = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)),('scaler', StandardScaler())])

    # Preprocessing for categorical data for most frequent
    categorical_transformer_mf = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False))])

    # Preprocessing for categorical data for constant
    categorical_transformer_c = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='NA')), ('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False))])


    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(transformers=[
        ('num_mean', numerical_transformer_m, mean_num_cols),
        ('num_constant', numerical_transformer_c, constant_num_cols),
        ('cat_mf', categorical_transformer_mf, mf_categorical_cols),
        ('cat_c', categorical_transformer_c, constant_categorical_cols)])


    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    print("shape of X_train before:")
    print(X_train.shape)
    print("shape of y_train before:")
    print(y_train.shape)
    train_data = np.concatenate([X_train, y_train], axis=0)
    valid_data = np.concatenate([X_test, y_test], axis=0)
    print("shape of X_train after:")
    print(X_train.shape)
    
    return train_data, valid_data

ws = Workspace.from_config()

print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

    
