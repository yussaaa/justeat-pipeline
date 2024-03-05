from io import StringIO
import pandas as pd
import numpy as np

def data_ingestion(SAMPLE_DATA: str) -> (pd.DataFrame, dict):

    df = pd.read_csv(SAMPLE_DATA)
    # df = pd.read_csv('final_dataset.csv')
    df.dropna(axis=0, inplace=True)

    #unique restaurants
    restaurants_ids = {}
    list_restaurants_ids = []
    for a,b in zip(df.restaurant_lat, df.restaurant_lon):
        id = "{}_{}".format(a,b)
        restaurants_ids[id] = {"lat": a, "lon":b}
    for i,key in enumerate(restaurants_ids.keys()):
        restaurants_ids[key]['id'] = i

    #labeling of restaurants
    df['restaurant_id']=[restaurants_ids["{}_{}".format(a,b)]['id'] for a,b in zip(df.restaurant_lat, df.restaurant_lon)]
    
    # # number of unique restaurants
    # len(restaurants_ids)
    return df, restaurants_ids
    