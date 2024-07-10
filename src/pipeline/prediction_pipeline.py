import os.path

import pandas as pd

from src.utils import load_pickle_file


class Features:
    def __init__(self,number_of_riders,number_of_drivers,location_category,customer_loyalty_status,
                 number_of_past_rides,average_ratings,time_of_booking,vehicle_type,expected_ride_duration):
         self.number_of_riders=number_of_riders
         self.number_of_drivers = number_of_drivers
         self.location_category = location_category
         self.customer_loyalty_status = customer_loyalty_status
         self.number_of_past_rides = number_of_past_rides
         self.average_ratings = average_ratings
         self.time_of_booking = time_of_booking
         self.vehicle_type = vehicle_type
         self.expected_ride_duration = expected_ride_duration
    def to_dataframe(self):
        try:
            feature_in_dict={
                'number_of_riders':[self.number_of_riders],
                'number_of_drivers':[self.number_of_drivers],
                'location_category':[self.location_category],
                'customer_loyalty_status':[self.customer_loyalty_status],
                'number_of_past_rides':[self.number_of_past_rides],
                'average_ratings':[self.average_ratings],
                'time_of_booking':[self.time_of_booking],
                'vehicle_type':[self.vehicle_type],
                'expected_ride_duration':[self.expected_ride_duration],
            }

            feature_in_df=pd.DataFrame(feature_in_dict)
            return feature_in_df
        except Exception as e:
            pass

class Prediction:
    def __init__(self):
        pass
    def initiate_prediction(self,features):
        preprocessing_path=os.path.join("artifacts/pickle","data_transformation.pkl")
        model_path=os.path.join("artifacts/pickle","model.pkl")

        preprocessing=load_pickle_file(preprocessing_path)
        model=load_pickle_file(model_path)

        processed_data = preprocessing.transform(features)
        result = model.predict(processed_data)

        return result