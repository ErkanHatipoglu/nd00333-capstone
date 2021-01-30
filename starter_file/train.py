from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import argparse
import os
import numpy as np

import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
# 
from azureml.core import Workspace, Dataset
from azureml.data.datapath import DataPath

def clean_data(data):

    # Remove rows with missing target, separate target from predictors
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X.SalePrice 

    # Remove target and 'Utilities' 
    X.drop(['SalePrice', 'Utilities'], axis=1, inplace=True)

    # Select object columns
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

    # Select numeric columns
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64','float64']]

    # Imputation lists

    # imputation to null values of these numerical columns need to be 'constant'
    constant_num_cols = ['GarageYrBlt', 'MasVnrArea']

    # imputation to null values of these numerical columns need to be 'mean'
    mean_num_cols = list(set(numerical_cols).difference(set(constant_num_cols)))

    # imputation to null values of these categorical columns need to be 'constant'
    constant_categorical_cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

    # imputation to null values of these categorical columns need to be 'most_frequent'
    mf_categorical_cols = list(set(categorical_cols).difference(set(constant_categorical_cols)))

    my_cols = constant_num_cols + mean_num_cols + constant_categorical_cols + mf_categorical_cols

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

    # Make pipeline
    my_pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
    # fit transform X
    X = my_pipeline.fit_transform (X, y)
    
    
    return X, y

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters
# https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train-hyperparameter-tune-deploy-with-sklearn/train_iris.py

web_path = ['https://github.com/ErkanHatipoglu/nd00333-capstone/tree/master/starter_file/data/train.csv']
ds = TabularDatasetFactory.from_delimited_files(path=web_path)


x, y = clean_data(ds)

# TODO: Split data into train and test sets.

### YOUR CODE HERE ###
x_train, x_test, y_train, y_test = train_test_split(x,y)
run = Run.get_context()

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate shrinks the contribution of each tree by learning_rate.")
    parser.add_argument('--est', type=int, default=100, help="The number of boosting stages to perform")

    args = parser.parse_args()

    run.log("Learning rate:", np.float(args.lr))
    run.log("Number of Estimators:", np.int(args.est))

    model = GradientBoostingRegressor(learning_rate = args.lr,           n_estimators=args.est).fit(x_train, y_train)


    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test,preds)  
    
    run.log("MAE", np.float(mae))

    # https://knowledge.udacity.com/questions/357007
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')


if __name__ == '__main__':
    main()