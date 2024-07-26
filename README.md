# Project: Dynamic Price Prediction
## Problem statement
### 1. Goal
```
To predict the price of ride 
```
### 2. About dynamic price
```requirements
Dynamic pricing is a strategy in which product or service prices continue to adjust in response to the real-time supply and demand (per Business Insider). It has filled the shortcomings of historical pricing strategies that were somewhat rigid in their structure and not easily responsive to changes in demand and supply.
```
## Description
### 1. Dataset
```
1. To predict the price of ride base on different scenrio such as if the the duration of ride increase how much it will effect on ride price and other factors like number of rider ,driver and etc
2. Dataset available on kaggle 
```

### 2. Features
``` 
Input features = [number_of_riders, number_of_drivers, location_category, customer_loyalty_status, number_of_past_rides,
                 average_ratings, time_of_booking, vehicle_type, expected_ride_duration]
Target feature = [historical_cost_of_ride]
```
### 3. Pipeline Structure
```requirements
Google define pipeline 
```
# Requirements
### 1. Language
```
Python 3.10
```
### 2. Libraries
```
1. numpy
2. pandas
3. scikit-learn
4. pickle
5. os 
6. streamlit 
 ```
# code
### 1. Enviroment
```requirements
conda create -p venv python==3.10 -y 
```
### 2.setup
```
The setup.py is a Python script typically included with Python-written libraries or apps. Its objective is to ensure that the program is installed correctly. 
```
### 3. Components
- Data ingestion
```
reading data from different source and splitting data into train and test
```
- Data transformation
```
  reading train and test dataset and apply different transformation and save transformation setting in pickle format
```
- Model training
```requirements
transformed dataset and using different machine learning model and save the best model in pickle format
```
### 4. Pipeline
- Training pipeline
```
using components and creating pipeline for model training
```
- Prediction pipeline
```
taking data from user transform for model and predict 
```

## Run
#### 1. Download repository
```
git clone https://github.com/ehetshamshaukat/dynamic_price.git
```
#### 2. Install dependences
```requirements
pip install -r requirements.txt
```
#### 3. Transformation and training
- data transformation and model training
  ```
  For model training which will also save tranformation and model in pickle format
  python src/pipeline/training_pipeline.py
  ```
- Prediction
  ```
  For Prediction use
  python src/pipeline/prediction_pipeline.py
  ```
#### 4. Streamlit
```
streamlit run application.py
```
## Deployment
```
Deploy on AWS using Github actions which is CI CD technique
```
## Image
<img width="1499" alt="Screenshot 2024-07-24 at 6 13 13 PM 1" src="https://github.com/user-attachments/assets/b76ef938-9cea-4d6a-bf1d-085f028bff6a">


