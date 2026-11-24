from src.data_ingestion import data_ingestion
from src.data_preprocessing import data_preprocessing
from src.model_building import model_build
from src.model_prediction import model_predict
from src.model_evaluation import model_evaluate


def main():
    # step 1: data ingestion
    df = data_ingestion()
    print(df.columns)


    # step 2: data preprocessing
    X_train_resample,X_test_transform,y_train_resample,y_test = data_preprocessing(df)
    

    # step 3: model build 

    model = model_build(X_train_resample,X_test_transform,y_train_resample,y_test)

    # step 4: Model Prediction 

    y_pred = model_predict(model,X_test_transform)

    # step 5: model evalute

    score =  model_evaluate(y_test,y_pred)

    print(f"Model Classification Report:{score}")

main()
