# model_trainer.py
import sys
from typing import Generator, List, Tuple
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    # artifact_folder constant is expected to come from src.constant
    artifact_folder = artifact_folder
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join("config", "model.yaml")


class ModelTrainer:
    def __init__(self):
        # small but necessary initialization so rest of code can use self.model_trainer_config
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()

        self.models = {
            "XGBClassifier": XGBClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "RandomForestClassifier": RandomForestClassifier(),
        }

    def evaluate_models(self, x, y, models: dict):
        """
        Trains each model on a train split and returns a dict of {model_name: test_score}
        """
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42
            )

            report = {}

            # iterate in a stable way over models
            for model_name, model in models.items():
                model.fit(x_train, y_train)

                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_model_score = accuracy_score(y_train, y_train_pred)
                test_model_score = accuracy_score(y_test, y_test_pred)

                report[model_name] = test_model_score

            return report
        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(
        self,
        x_train: np.array,
        y_train: np.array,
        x_test: np.array,
        y_test: np.array,
    ):
        """
        Optional helper — keeps original signature but uses evaluate_models to compare models.
        """
        try:
            # evaluate_models expects x,y,models — we pass training set for evaluation (it does internal split)
            model_report: dict = self.evaluate_models(x_train, y_train, self.models)

            print(model_report)

            # choose best by test score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model_object = self.models[best_model_name]

            return best_model_name, best_model_object, best_model_score

        except Exception as e:
            raise CustomException(e, sys)

    def finetune_best_model(
        self,
        best_model_object: object,
        best_model_name: str,
        x_train,
        y_train,
    ) -> object:
        """
        Read param grid from config/model.yaml through utils and return a model with best params applied.
        """
        try:
            
            config = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)
            model_param_grid = config["model_selection"]["model"][best_model_name]["search_param_grid"]

            grid_search = GridSearchCV(
                estimator=best_model_object,
                param_grid=model_param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(x_train, y_train)

            best_params = grid_search.best_params_
            print("best params are :", best_params)

            finetuned_model = best_model_object.set_params(**best_params)

            return finetuned_model
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        """
        Main entry: expects train_array and test_array with last column = label.
        Returns path to saved model.
        """
        try:
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            logging.info("Extracting model config file path")

            # evaluate models using training data
            model_report: dict = self.evaluate_models(x_train, y_train, self.models)

            # to get best model score
            best_model_score = max(model_report.values())

            # to get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = self.models[best_model_name]

            # finetune the best model using the training set
            best_model = self.finetune_best_model(
                best_model_object=best_model,
                best_model_name=best_model_name,
                x_train=x_train,
                y_train=y_train,
            )

            # train finetuned model on full training set and evaluate on test set
            best_model.fit(x_train, y_train)
            y_pred = best_model.predict(x_test)
            best_model_score = accuracy_score(y_test, y_pred)

            print(f"best model name {best_model_name} and score: {best_model_score}")

            # threshold message: keep numbers consistent
            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No best model found with an accuracy greater than the threshold {self.model_trainer_config.expected_accuracy}"
                )
            logging.info("Best found model on both training and testing dataset")

            logging.info(f"Saving model at path :{self.model_trainer_config.trained_model_path}")

            
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_path), exist_ok=True)

            # save model using utils (MainUtils.save_object expected)
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)







            
                

                    
            


                
    