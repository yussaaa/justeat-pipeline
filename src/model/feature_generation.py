import collections.abc
from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd
import h3
from sklearn.preprocessing import LabelEncoder



def calc_dist(p1x, p1y, p2x, p2y):
  p1 = (p2x - p1x)**2
  p2 = (p2y - p1y)**2
  dist = np.sqrt(p1 + p2)
  return dist.tolist() if isinstance(p1x, collections.abc.Sequence) else dist

# calc. avg. distance to restaurants
def avg_dist_to_restaurants(courier_lat,courier_lon, restaurants_ids):
  return np.mean([calc_dist(v['lat'], v['lon'], courier_lat, courier_lon) for v in restaurants_ids.values()])

def calc_haversine_dist(lat1, lon1, lat2, lon2):

  R = 6372.8    #3959.87433  this is in miles.  For Earth radius in kilometers use 6372.8 km
  if isinstance(lat1, collections.abc.Sequence):
    dLat = np.array([radians(l2 - l1) for l2,l1 in zip(lat2, lat1)])
    dLon = np.array([radians(l2 - l1) for l2,l1 in zip(lon2, lon1)])
    lat1 = np.array([radians(l) for l in lat1])
    lat2 = np.array([radians(l) for l in lat2])
  else:
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

  a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
  c = 2*np.arcsin(np.sqrt(a))
  dist = R*c
  return dist.tolist() if isinstance(lon1, collections.abc.Sequence) else dist

# calc. avg. distance to restaurants
def avg_Hdist_to_restaurants(courier_lat,courier_lon, restaurants_ids):
  return np.mean([calc_haversine_dist(v['lat'], v['lon'], courier_lat, courier_lon) for v in restaurants_ids.values()])



def feature_distance_creation(df, restaurants_ids):
    df['dist_to_restaurant'] = calc_dist(df.courier_lat, df.courier_lon, df.restaurant_lat, df.restaurant_lon)
    df['avg_dist_to_restaurants'] = [avg_dist_to_restaurants(lat,lon) for lat,lon in zip(df.courier_lat, df.courier_lon)]
    df['Hdist_to_restaurant'] = calc_haversine_dist(df.courier_lat.tolist(), df.courier_lon.tolist(), df.restaurant_lat.tolist(), df.restaurant_lon.tolist())
    df['avg_Hdist_to_restaurants'] = [avg_Hdist_to_restaurants(lat,lon) for lat,lon in zip(df.courier_lat, df.courier_lon)]
    return df
    

#STEP 1 - define K & initiate data
def initiate_centroids(k, df):
    '''
    Select k data points as centroids
    k: number of centroids
    dset: pandas dataframe
    '''
    centroids = df.sample(k)
    return centroids

# df_couriers = pd.DataFrame({})
# df_couriers['lat'] = df['courier_lat']
# df_couriers['lon'] = df['courier_lon']


# STEP 2 - define distance metric : Euclidean distance
## TODO: clean this in the end
def eucl_dist(p1x,p1y,p2x,p2y):
  return calc_dist(p1x, p1y, p2x, p2y)

# STEP 3 - Centroid assignment
def centroid_assignation(df, centroids):
  k = len(centroids)
  n = len(df)
  assignation = []
  assign_errors = []
  centroids_list = [c for i,c in centroids.iterrows()]
  for i,obs in df.iterrows():
    # Estimate error
    all_errors = [eucl_dist( centroid['lat'],
                            centroid['lon'],
                            obs['courier_lat'],
                            obs['courier_lon']) for centroid in centroids_list]

    # Get the nearest centroid and the error
    nearest_centroid =  np.where(all_errors==np.min(all_errors))[0].tolist()[0]
    nearest_centroid_error = np.min(all_errors)

    # Add values to corresponding lists
    assignation.append(nearest_centroid)
    assign_errors.append(nearest_centroid_error)
  df['Five_Clusters_embedding'] =assignation
  df['Five_Clusters_embedding_error'] =assign_errors
  return df



def assign_centroids_to_df(df, restaurants_ids):

    np.random.seed(1)
    k=5
    df_restaurants = pd.DataFrame([{"lat": v['lat'], "lon": v['lon']} for v in restaurants_ids.values()])
    centroids = initiate_centroids(k, df_restaurants)

    df = centroid_assignation(df,centroids)
    return df


def h3_features(df, resolution=7):
    
    df['courier_location_timestamp']=  pd.to_datetime(df['courier_location_timestamp'])
    df['order_created_timestamp'] = pd.to_datetime(df['order_created_timestamp'])
    df['h3_index'] = [h3.geo_to_h3(lat,lon,resolution) for (lat,lon) in zip(df.courier_lat, df.courier_lon)]
    df['date_day_number'] = [d for d in df.courier_location_timestamp.dt.day_of_year]
    df['date_hour_number'] = [d for d in df.courier_location_timestamp.dt.hour]

    index_list = [(i,d,hr) for (i,d,hr) in zip(df.h3_index, df.date_day_number, df.date_hour_number)]
    set_indexes = list(set(index_list))
    dict_indexes = {label: index_list.count(label) for label in set_indexes}
    df['orders_busyness_by_h3_hour'] = [dict_indexes[i] for i in index_list]

    restaurants_counts_per_h3_index = {a:len(b) for a,b in zip(df.groupby('h3_index')['restaurant_id'].unique().index, df.groupby('h3_index')['restaurant_id'].unique()) }
    df['restaurants_per_index'] = [restaurants_counts_per_h3_index[h] for h in df.h3_index]

    return df 


def Encoder(df):
  columnsToEncode = list(df.select_dtypes(include=['category','object']))
  le = LabelEncoder()
  for feature in columnsToEncode:
      try:
          df[feature] = le.fit_transform(df[feature])
      except:
          print('Error encoding '+feature)
  return df

def generate_features(df, restaurants_ids):
    df = feature_distance_creation(df, restaurants_ids)
    df = assign_centroids_to_df(df)
    df = h3_features(df)
    df['h3_index'] = df.h3_index.astype('category')
    df = Encoder(df)
    return df 




