import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from pandas import DataFrame


class MultipleTime:
    def __init__(self, counting: DataFrame, segments: GeoDataFrame, nodes: GeoDataFrame, fields_to_use: list):
        self.counting = counting[fields_to_use]
        print(self.counting.columns)
        pd.Timestamp(tz='Europe/Berlin', year=2019, month=5, day=23, hour=6, minute=15)
