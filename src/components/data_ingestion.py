import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import os


@dataclass
class DataIngestionConfig:
    train_dataset_path = os.path.join("artifacts/train_test_dataset", "train.csv")
    test_dataset_path = os.path.join("artifacts/train_test_dataset", "test.csv")


class DataIngestion:
    def __init__(self):
        self.dataset = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # reading dataset
            path = ("dataset/dynamic_pricing.csv")
            df = pd.read_csv(path)

            # making directory for train N test dataset
            os.makedirs(os.path.dirname(self.dataset.train_dataset_path), exist_ok=True)

            # splitting training and testing dataset
            train_dataset, test_dataset = train_test_split(df, test_size=0.2, random_state=69)

            # saving train N test dataset
            train_dataset.to_csv(self.dataset.train_dataset_path,header=True,index=False)
            test_dataset.to_csv(self.dataset.test_dataset_path,header=True,index=False)

            # returning train N test dataset path for data transformation
            return self.dataset.train_dataset_path, self.dataset.test_dataset_path
        except Exception as e:
            raise e
