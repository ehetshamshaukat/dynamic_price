import os
from src.utils import save_file_as_pickle
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score
from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    model_pickle_path=os.path.join("artifacts/pickle","model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_config=ModelTrainingConfig()

    def initiate_model_training(self,transform_train_dataset,transform_test_dataset):
        try:
            xtrain=transform_train_dataset[:,:-1]
            ytrain=transform_train_dataset[:,-1]

            xtest=transform_test_dataset[:,:-1]
            true_value=transform_test_dataset[:,-1]

            models={
                "linear_regression":LinearRegression(),
                "stochastic_regression":SGDRegressor(),
                "svr":SVR(),
                "decision_tree":DecisionTreeRegressor(),
                "Random_forest":RandomForestRegressor(),
                "Ada_boost":AdaBoostRegressor(),
                "Gradient_boost":GradientBoostingRegressor()
            }
            model_report={}
            for model_name,model in models.items():
                model.fit(xtrain,ytrain)
                predicted_value=model.predict(xtest)
                r2=r2_score(true_value,predicted_value)
                model_report[model_name]=r2*100

            best_model_name=max(model_report,key=model_report.get)
            best_model=models[best_model_name]
            save_file_as_pickle(self.model_config.model_pickle_path,best_model)
        except Exception as e:
            raise e