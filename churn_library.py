'''
Churn credit analysis

Author: Julián David Pérez Hincapié
'''


# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    # read the data set
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    # Perform an exploratory analysis about the dataset
    print('Head to the dataset: \n{}'.format(df.head()))
    print('Shape of the dataset: {}'.format(df.shape))
    print('There are null data?: \n{}'.format(df.isnull().sum()))

    # Perform graphical analysis to the data
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist()
    plt.savefig(r'images/eda/eda_churn_hist.png')

    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist()
    plt.savefig(r'images/eda/eda_Customer_Age.png')

    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(r'images/eda/eda_Marital_Status.png')

    plt.figure(figsize=(20,10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig(r'images/eda/eda_Total_Trans_Ct.png')

    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig(r'images/eda/eda_heatmap.png')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    # Perform the encoding to the dataset
    for category in category_lst:
        category_list=[]
        category_group=df.groupby(category).mean()[response]

        for val in df[category]:
                category_list.append(category_group.loc[val])

        new_column=category+"_"+response
        
        df[new_column] = category_list

    return df


def perform_feature_engineering(df,  response, remove_list=[],):
    '''
    Function that allow to remove variables and do the train test split variables

    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]
              remove_list: list of variables that will be remove to the list of X_train and X_test

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Set response variable and Dataframe
    y = df[response]
    X= pd.DataFrame()

    # Set what variables will be the importants variables to the dataset X
    num_cols = df._get_numeric_data().columns
    keep_cols=[]
  
    for col in num_cols:
        keep_cols.append(col)

    for col in remove_list:
        keep_cols.remove(col)

    X[keep_cols] = df[keep_cols]

    # train test split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores about the models and save as csv
    print('random forest results')
    print('test results')
    classification_report_rdf=classification_report(y_test, y_test_preds_rf)
    print(classification_report_rdf)

    print('train results')
    classification_report_rdf=classification_report(y_train, y_train_preds_rf)
    print(classification_report_rdf)


    print('logistic regression results')
    print('test results')
    classification_report_lg=classification_report(y_test, y_test_preds_lr)
    print(classification_report_lg)

    print('train results')
    classification_report_lg=classification_report(y_train, y_train_preds_lr)
    print(classification_report_lg)


    # Save classification report as image   
    plt.figure(figsize=(20,10)) 
    plt.rc('figure', figsize=(5, 5))
    #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r'images/results/rdf_classification_report.png')

    plt.figure(figsize=(20,10))
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r'images/results/lr_classification_report.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # PLot and save summary_plot
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar",show=False)
    image_path=output_pth+'/summary_plot.png'
    plt.savefig(image_path)

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,10))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    image_path=output_pth+'/Feature_importance.png'
    plt.savefig(image_path)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    # Define the parameters
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }

    # Train the models Random forest and logistic regression
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save the Classification report of the models 
    classification_report_image(y_train,y_test,y_train_preds_lr,y_train_preds_rf,y_test_preds_lr,y_test_preds_rf)

    # Save the ROC curves to evaluate the models
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(r'images/results/ROC_curve_models.png')

    # Store the traind models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Explain importance of variables in random forest
    feature_importance_plot(cv_rfc,X_test,'images/results')


def classification_report_csv(report, name_model):
    '''
        Function that allows to save the classification report of the models in csv

        input:
                report: classification_report type of sk learn
                name_model: str with the name of the model evaluate
        output:
                None
    
    '''
    # Generating the csv file
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe_result = pd.DataFrame.from_dict(report_data)
    path_name = 'images/results/classification_report_'+name_model+'.csv'
    dataframe_result.to_csv(path_name, index = False)


if __name__ == "__main__":
    
    data_frame=import_data(r"./data/bank_data.csv")
    perform_eda(data_frame)
    CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
    ]
    data_frame=encoder_helper(data_frame,CAT_COLUMNS,'Churn')
    REMOVE_COLS = ['Unnamed: 0','CLIENTNUM','Churn']
    X_train,X_test,y_train,y_test=perform_feature_engineering(data_frame,'Churn',REMOVE_COLS)
    train_models(X_train,X_test,y_train,y_test)

