import pandas as pd
from src.features.build_features import (
    get_geo_coordinates,
    process_data,
    add_nearest_metro_features,
    add_kremlin_distance_feature,
    add_walk_time_to_metro_feature,
    process_df,
    reduce_mem_usage,
)

KREMLIN_COORDS = (55.7520, 37.6175)
AVERAGE_WALK_SPEED = 5

FEATURE2DROP = [
    "index_right",
    "lat",
    "osm_transport_stop_points_in_0.01",
    "lng",
    "osm_crossing_points_in_0.01",
]


def json_to_dataframe(json_data: dict) -> pd.DataFrame:
    data: dict = json_data
    df: pd.DataFrame = pd.DataFrame([data])
    return df


def add_features(df: pd.DataFrame, geo, stations) -> pd.DataFrame:
    df = (
        df.pipe(get_geo_coordinates).pipe(
            process_data,
            geo=geo).pipe(
            add_nearest_metro_features,
            stations=stations).pipe(
                add_kremlin_distance_feature,
                kremlin_coords=KREMLIN_COORDS).pipe(
                    add_walk_time_to_metro_feature,
                    average_walk_speed=AVERAGE_WALK_SPEED) .pipe(process_df).drop(
                        FEATURE2DROP,
            axis=1) .pipe(reduce_mem_usage))
    return df
