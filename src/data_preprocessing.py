
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer,KNNImputer 
from imblearn.over_sampling import SMOTE


def data_preprocessing(df):
        # step 1: drop the duplicates values 
        # step 2; feacture engineering (Optional)
        # step 3: split the dataset into X and y 
        # step 4: split the dataset into train and test 
        # step 5: Segregate the data into numerical and categorical data
        # step 6: Build numerical_pipeline :imputation and scaling 
        # step 7: Build categorical_pipeline : imputation and encoding
        # step 8: Concat numerical and categorical pipeline and columns transformer
        # step 9: fit the transformer on the train and test 
        #step 10: fit_resample the data so that target columnns balance (only for the categoricsl columns)
        # step 11: Return the preprocessing X_train, X_test, y_train, y_test
        #--------------------------------------------------------------------------------------------------------

        # step 1: drop the duplicates values
        df.drop_duplicates(inplace=True)

        # step 2: features engineering (Optional)
        # step 3: split the dataset into X and y

        X = df.drop(columns = ["customerID","Churn"])
        y = df["Churn"]

            # step 4: split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                                random_state = 1 )

        # step 5: Segregate the data into numerical and categorical data

        numerical_col = X_train.select_dtypes(exclude = "object").columns
        categorical_col = X_train.select_dtypes(include = "object").columns


        # step 6: Build numerical_pipeline :imputation and scaling

        numerical_pipeline = Pipeline([("imputation",SimpleImputer(strategy = "median")),
                                      ("scaling", MinMaxScaler())])

        categorical_pipeline = Pipeline([("imputation",SimpleImputer(strategy = "most_frequent")),
                                         ("encoding", OneHotEncoder(handle_unknown = "ignore"))])

        # step 8: Concat numerical and categorical pipeline and columns transformer

        transformer = ColumnTransformer([("numerical_pipeline", 
                                        numerical_pipeline,
                                         numerical_col),

                                          ("categorical_pipeline",
                                            categorical_pipeline,
                                              categorical_col)])


    #step 9: fit the transformer on the train and test
    
        X_train_transform = transformer.fit_transform(X_train) # seen data
        X_test_transform = transformer.transform(X_test) # unseen data

    #step 10: fit_resample the data so that target columnns balance (only for the categoricsl columns)
    
        '''
        * Generally used with classification model 
        * Target columns balance technique basically classified into two 

        ---> over_sampling Technique

        ---> under_sampling Technique

        '''
        sm = SMOTE(random_state = 1)
        X_train_resample,y_train_resample = sm.fit_resample(X_train_transform,y_train)

        #step 11: Return the preprocessing X_train, X_test, y_train, y_test

        return X_train_resample,X_test_transform,y_train_resample,y_test





