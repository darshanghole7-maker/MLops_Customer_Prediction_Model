import pandas as pd 

def data_ingestion():

    '''
    * Independent Columns(X): 'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges',


    * Target Columns(y): "churn"

    * Problem Statments 
    1. To predict the churn rate 
    2. Use Clustering approach to predict and classify the cluster pattern such as {11,12 & 13}

    '''


    df =pd.read_csv("C:\\MLops_Customer_Prediction_Model\\data\\raw\\churn - churn.csv")

    return df 



