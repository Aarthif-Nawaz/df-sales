from pandas import DataFrame
import pickle
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd


filename = 'xgb_reg.pkl'


def create_features(df, label=None):
    X = df.loc[:, df.columns != 'Sales_Amount']
    if label:
        y = df[label]
        return X, y
    return X


# Feature selection
def select_features(X_train, y_train, X_test):
    # Configure to select a subset of features
    fs = SelectFromModel(xgb.XGBRegressor(), threshold=0.00005)

    # Learn relationship from training data
    fs.fit(X_train, y_train)

    # Transform train input data
    X_train_fs = fs.transform(X_train)

    # Transform test input data
    X_test_fs = fs.transform(X_test)

    return X_train_fs, X_test_fs, fs


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def mean_scale(df, column_to_scale):
    mean = np.mean(df[column_to_scale])
    df[column_to_scale] = df[column_to_scale] / mean
    return mean


sap_data_df = pd.read_excel('글리아타민_data_NUMERICAL.xlsx')

# Rename 'DayOfMonth' column to 'Day'
sap_data_df.rename(columns={'DayOfMonth': 'Day'}, inplace=True)

# Sort dataframe
sap_data_df_sorted = sap_data_df.sort_values(by=['Year', 'Month', 'Day'], ignore_index=True)

# Remove points that are >= 0.5e+13 (0.5 * 10^13)
sap_data_df_sorted_new = sap_data_df_sorted.loc[sap_data_df_sorted['Sales_Amount'] < 0.5e+13]
sap_data_df_sorted_new['Sales_Amount'].describe().apply(lambda x: format(x, 'f'))

# Split date for training and testing set
split_date = '2020-09-01'

# Get train and test data

df_train = sap_data_df_sorted_new.loc[
    pd.to_datetime(sap_data_df_sorted_new[['Year', 'Month', 'Day']]) <= pd.to_datetime(split_date)].reset_index(
    drop=True)
df_test = sap_data_df_sorted_new.loc[
    pd.to_datetime(sap_data_df_sorted_new[['Year', 'Month', 'Day']]) > pd.to_datetime(split_date)].reset_index(
    drop=True)

# # Get features and labels
X_train, y_train = create_features(df_train, label='Sales_Amount')
X_test, y_test = create_features(df_test, label='Sales_Amount')

# Select features
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)


def Training():
    # A parameter grid for XGBoost
    params = {
        'min_child_weight': [3, 4, 5],
        'gamma': [i / 10.0 for i in range(3, 6)],
        'subsample': [i / 10.0 for i in range(6, 11)],
        'colsample_bytree': [i / 10.0 for i in range(6, 11)],
        'max_depth': [2, 3, 4]
    }

    # Initialize XGB model and GridSearch
    xgb_reg = xgb.XGBRegressor(nthread=-1, objective='reg:squarederror')
    grid = GridSearchCV(xgb_reg, params)

    # Fit training data
    grid.fit(X_train_fs, y_train)

    gridcv_xgb = grid.best_estimator_
    pickle.dump(gridcv_xgb, open(filename, "wb"))


############################TESTING PART#####################

def testing(file):
    xgb_model_loaded = pickle.load(open('xgb_reg.pkl', "rb"))
    testing = pd.read_csv(f'Files/{file}')
    X_train_fs, X_testing_fs, fs = select_features(X_train, y_train, testing)
    # Display results of XGBoost model
    testing['Prediction'] = xgb_model_loaded.predict(X_testing_fs)
    return testing['Prediction']


def testingIndividual(day, month, year, quantity):
    data = [[day, month, year, quantity]]
    xgb_model_loaded = pickle.load(open('xgb_reg.pkl', "rb"))
    df = DataFrame(data, columns=['Day', 'Month', 'Year', 'Quantity'])
    X_train_fs, X_testing_fs, fs = select_features(X_train, y_train, df)
    df['Prediction'] = xgb_model_loaded.predict(X_testing_fs)
    print(df.head())
    return df['Prediction']

