import pandas as pd
import geopy
import numpy as np
from geopy.geocoders import Nominatim
import geopandas as gpd
from pandas import DataFrame
from scipy.spatial import cKDTree
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import warnings

warnings.filterwarnings("ignore")

KREMLIN_COORDS = (55.7520, 37.6175)
AVERAGE_WALK_SPEED = 5


def get_lat_long(region, street, house_number):
    geolocator = Nominatim(user_agent="your_app_name")
    address = f"{region}, {street}, {house_number}"
    location = geolocator.geocode(address, country_codes="RU", language="ru")
    if location:
        return location.latitude, location.longitude
    else:
        return None, None


def get_geo_coordinates(data: pd.DataFrame) -> pd.DataFrame:
    data["geo_lat"], data["geo_lon"] = zip(
        *data.apply(lambda row: get_lat_long(row["city"], row["street"], row["house_number"]), axis=1))
    return data


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ["object", "category", "datetime64[ns, UTC]"]:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


def process_data(data: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    def create_geodataframe(df: pd.DataFrame, lon_col: str, lat_col: str) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col]))

    def buffer_geodataframe(gdf: gpd.GeoDataFrame, buffer_distance: float) -> gpd.GeoDataFrame:
        gdf_buffered = gdf.copy()
        gdf_buffered['geometry'] = gdf_buffered['geometry'].buffer(buffer_distance)
        return gdf_buffered

    def join_geodataframes(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, batch_size: int) -> pd.DataFrame:
        gdf_joined_list = []
        for i in range(0, len(gdf1), batch_size):
            batch = gdf1.iloc[i:i + batch_size]
            batch_joined = gpd.sjoin_nearest(batch, gdf2, how='left')
            batch_joined = batch_joined.drop_duplicates(
                subset=["geo_lat", "geo_lon", "region", "building_type", "area"], keep="last")
            gdf_joined_list.append(batch_joined)
        return pd.concat(gdf_joined_list, ignore_index=True)

    data_geo = create_geodataframe(data, "geo_lon", "geo_lat")
    geo_geo = create_geodataframe(geo, "lng", "lat")
    geo_geo = geo_geo.drop(["city"], axis=1)

    gdf1_buffered = buffer_geodataframe(data_geo, 0.02)
    batch_size = 1000
    data_joined = join_geodataframes(gdf1_buffered, geo_geo, batch_size)
    data_joined = data_joined.drop("geometry", axis=1)

    return data_joined


def create_metro_tree(stations: DataFrame) -> cKDTree:
    return cKDTree(stations[["geo_lat", "geo_lon"]])


def find_nearest_station(metro_tree: cKDTree, row: DataFrame, stations: DataFrame) -> str:
    house_location = (row["geo_lat"], row["geo_lon"])
    _, nearest_station_index = metro_tree.query(house_location)
    return stations.loc[nearest_station_index, "Station"]


def create_ball_tree(stations: DataFrame) -> BallTree:
    metro_coordinates = stations[["geo_lat", "geo_lon"]].values
    return BallTree(metro_coordinates, leaf_size=40, metric='haversine')


def find_nearest_metro(ball_tree: BallTree, row: DataFrame, stations: DataFrame) -> float:
    point = (row['geo_lat'], row['geo_lon'])
    _, indices = ball_tree.query([point], k=1)
    nearest_metro_coords = stations[["geo_lat", "geo_lon"]].values[indices[0][0]]
    distance = geodesic(point, nearest_metro_coords).km
    return distance


def add_nearest_metro_features(data: DataFrame, stations: DataFrame) -> DataFrame:
    metro_tree = create_metro_tree(stations)
    data["nearest_metro_station"] = data.apply(lambda row: find_nearest_station(metro_tree, row, stations),
                                               axis=1)
    ball_tree = create_ball_tree(stations)
    data['metro_dist'] = data.apply(lambda row: find_nearest_metro(ball_tree, row, stations), axis=1)

    return data


def calculate_kremlin_distance(row: DataFrame, kremlin_coords: tuple) -> float:
    return geopy.distance.geodesic((row.geo_lat, row.geo_lon), kremlin_coords).km


def add_kremlin_distance_feature(data: DataFrame, kremlin_coords: tuple) -> DataFrame:
    data['kremlin_dist'] = data.apply(lambda row: calculate_kremlin_distance(row, kremlin_coords), axis=1)
    return data


def calculate_walk_time_to_metro(row: DataFrame, average_walk_speed: float) -> float:
    return (row.metro_dist / average_walk_speed) * 60


def add_walk_time_to_metro_feature(data: DataFrame, average_walk_speed: float) -> DataFrame:
    data['walk_time_to_metro'] = data.apply(lambda row: calculate_walk_time_to_metro(row, average_walk_speed), axis=1)
    return data


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    df['many_offices'] = (df['osm_offices_points_in_0.01'] > 10).astype(int)
    df['many_food'] = (df['osm_catering_points_in_0.01'] > 10).astype(int)
    df['many_shops'] = (df['osm_shops_points_in_0.01'] > 10).astype(int)
    df['many_financial_organizations'] = (df['osm_finance_points_in_0.01'] > 10).astype(int)
    df['many_entertainment'] = (df['osm_leisure_points_in_0.01'] > 1).astype(int)
    df['many_historical_objects'] = (df['osm_historic_points_in_0.01'] > 1).astype(int)
    df['many_hotels'] = (df['osm_hotels_points_in_0.01'] > 2).astype(int)
    df['station_rate'] = df['osm_train_stop_closest_dist'].apply(lambda x: 0 if x > 4 else 5 if 1 < x <= 4 else 1)
    df['many_stations'] = (df['osm_train_stop_points_in_0.01'] > 2).astype(int)
    df['many_culture_objects'] = (df['osm_culture_points_in_0.01'] > 2).astype(int)
    df['many_comfort_objects'] = (df['osm_amenity_points_in_0.01'] > 5).astype(int)
    df["level_to_levels"] = df["level"] / df["levels"]
    df["area_to_rooms"] = df["area"] / df["rooms"]
    return df
