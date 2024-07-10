import time
from src.components.data_ingestion import DataIngestion


if __name__ == "__main__":

    # data ingestion
    di=DataIngestion()
    print("*"*148)
    print("\t\t\t\t\t\t\tData Ingestion Starts")
    time.sleep(1)
    train_dataset_path,test_dataset_path=di.initiate_data_ingestion()
    time.sleep(1)
    print("\t\t\t\t\t\t\tData Ingestion Complete")

    # data transformation


