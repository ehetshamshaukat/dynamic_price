import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os
from dataclasses import dataclass
from src.utils import save_file_as_pickle


@dataclass
class DataTransformationConfig:
    data_transformation_pickle_path = os.path.join("artifacts/pickle", "data_transformation.pkl")


class DataTransformation:
    def __init__(self):
        self.data_Transformation_pickle = DataTransformationConfig()

    def data_transformation(self):
        try:
            numerical_columns = ['number_of_riders', 'number_of_drivers', 'number_of_past_rides',
                                 'average_ratings', 'expected_ride_duration']
            categorical_columns = ['location_category', 'customer_loyalty_status', 'time_of_booking', 'vehicle_type']
            numerical_column_pipeline = Pipeline(steps=[
                ("simpleImputer",SimpleImputer(strategy="mean")),
                ("standardscaler", StandardScaler())
            ])
            categorical_column_pipeline = Pipeline(steps=[
                ("simpleImputer",SimpleImputer(strategy="most_frequent")),
                ("ordinalencoder",OrdinalEncoder()),
                ("standardscaler", StandardScaler())
            ])

            processing = ColumnTransformer([
                ("numerical_column_pipeline", numerical_column_pipeline, numerical_columns),
                ("categorical_column_pipeline", categorical_column_pipeline, categorical_columns)
            ])
            return processing

        except Exception as e:
            raise e

    def initiate_data_transformation(self, train_dataset_path, test_dataset_path):
        try:
            train_dataset = pd.read_csv(train_dataset_path)
            test_dataset = pd.read_csv(test_dataset_path)

            train_dataset.columns = train_dataset.columns.str.lower()
            test_dataset.columns = test_dataset.columns.str.lower()

            column_to_drop = "historical_cost_of_ride"
            target_column = "historical_cost_of_ride"

            xtrain = train_dataset.drop(columns=column_to_drop, axis=1)
            ytrain = train_dataset[target_column]

            xtest = test_dataset.drop(columns=column_to_drop, axis=1)
            ytest = test_dataset[target_column]

            # loading and saving file in pickle format
            transform = self.data_transformation()
            save_file_as_pickle(self.data_Transformation_pickle.data_transformation_pickle_path, transform)

            # transforming data
            transform_xtrain = transform.fit_transform(xtrain)
            transform_xtest = transform.transform(xtest)

            # concatenation of transform data with y feature
            transform_train_dataset = np.c_[transform_xtrain, np.array(ytrain)]
            transform_test_dataset = np.c_[transform_xtest, np.array(ytest)]

            # returning transformed dataset
            return transform_train_dataset, transform_test_dataset

        except Exception as e:
            raise e
