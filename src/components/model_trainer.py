import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1],
                test_array[:, -1]
            )

            logging.info("Data split completed")

            models = {
                "Linear Regression":LinearRegression(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Adaboost Regressor":AdaBoostRegressor(),
                "KK neighbours":KNeighborsRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoost Regressor":CatBoostRegressor(verbose=0)
            }

            logging.info("Model eveluation initiated")

            model_report:dict = evaluate_models(X_train = X_train, y_train= y_train, 
                            X_test=X_test, y_test=y_test, models=models)
            logging.info("Model evaluation completed")
            
            # To get the best model score from the dict
            best_model_score = max(sorted(model_report.values()))
            logging.info(f"Best model score is {best_model_score}")

            # To get the best model name from the dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            logging.info(f"Best model name is {best_model_name}")

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                logging.info("NO best model found")
                raise CustomException("Best model score is less than 0.6", sys)

            logging.info("Best model score is greater than 0.6")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Best model saved")
            

            logging.info("Predicting the test data using best model")
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 square value is {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
