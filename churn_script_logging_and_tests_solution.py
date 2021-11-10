'''
	Churn customer credit test and login analysis

	Author: Julián David Pérez Hincapié

    Creation Date: 11/07/2021
'''

import logging
from time import gmtime, strftime
import churn_library_solution as cls


time_now = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

logging.basicConfig(
    filename='./logs/churn_library' + time_now + '.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions

input:
    import_data: This is a function to return a load pd dataframe
    '''
    try:
        logging.info("Start Testing import_data: Try to import data...")
        dataframe = import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function

input:
    perform_eda: This is a function to return nothing, but it do it a Eda analysis in the data
    '''

    try:
        logging.info("Start Testing perform_eda: Try to import data and perform eda...")
        dataframe = cls.import_data("./data/bank_data.csv")
        perform_eda(dataframe)
        logging.info("Testing perform_eda: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing perform_eda: The file wasn't found")
        raise err
    except NameError as err:
        logging.error(
            "Testing perform_eda: The file dataframe wasn't defined accuratly")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper

input:
    encoder_helper: this function return a dataframe with some encoded columns
    '''

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    try:
        logging.info("Start Testing encoder_helper: Try to import data and encoder variables of it...")
        dataframe = cls.import_data("./data/bank_data.csv")
        dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        dataframe = encoder_helper(dataframe, cat_columns, 'Churn')
    except FileNotFoundError as err:
        logging.error("Testing encoder_helper: The file wasn't found")
        raise err
    except NameError as err:
        logging.error(
            "Testing encoder_helper: The file dataframe wasn't defined accuratly")
        raise err
    except KeyError as err:
        logging.error(
            "Testing encoder_helper:  Any of the keys in cat_columns is bad defined")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The file doesn't appear to have rows and columns")
        raise err

    return dataframe


def test_perform_feature_engineering(
        perform_feature_engineering,
        dataframe_encoded):
    '''
    test perform_feature_engineering

input:
    perform_feature_engineering: Function that perform engineering about the data and
    make the separation of train and test variables

    dataframe_encoded: This is the output of encoder_helper, is necessary to have all
    the necessary data to process the ml

    '''

    remove_cols = ['Unnamed: 0', 'CLIENTNUM', 'Churn']

    try:
        logging.info("Start Testing test_perform_feature_engineering: Try to perform feature engineering...")
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dataframe_encoded, 'Churn', remove_cols)
    except FileNotFoundError as err:
        logging.error(
            "Testing test_perform_feature_engineering: The file wasn't found")
        raise err
    except NameError as err:
        logging.error(
            "Testing test_perform_feature_engineering: The file dataframe wasn't defined accuratly")
        raise err
    except KeyError as err:
        logging.error(
            "Testing test_perform_feature_engineering:  any of the keys in REMOVE_COLS is bad defined")
        raise err

    try:
        assert x_train.shape[0] > 0
        assert y_train.shape[0] > 0
        logging.info("Testing test_perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing test_perform_feature_engineering: The file doesn't appear to have rows and columns")
        raise err

    return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models

input:
    train_models: Function that train two models to predict customer churn
    X_train: X training data obtain from test_perform_feature_engineering
    X_test: X testing data obtain from test_perform_feature_engineering
    y_train: y training data obtain from test_perform_feature_engineering
    y_test: y testing data obtain from test_perform_feature_engineering
    '''

    try:
        logging.info("Start Testing test_train_models: Try to train model...")
        train_models(x_train, x_test, y_train, y_test)
        logging.info("Testing test_train_models: SUCCESS")
    except ValueError as err:
        logging.error(
            "Testing test_train_models: Some of the Train variables is empty or bad defined")
        raise err
    except NameError as err:
        logging.error(
            "Testing test_train_models: some of the train or test variables wasn't defined accuratly")
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    dataframe_encoded_result = test_encoder_helper(cls.encoder_helper)
    x_train_result, x_test_result, y_train_result, y_test_result = test_perform_feature_engineering(
        cls.perform_feature_engineering, dataframe_encoded_result)
    test_train_models(cls.train_models, x_train_result, x_test_result, y_train_result, y_test_result)
