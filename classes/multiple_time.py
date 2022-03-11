import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from pandas import DataFrame


class MultipleTime:
    def __init__(self, counting: DataFrame, segments: GeoDataFrame, nodes: GeoDataFrame, fields_to_use: list):
        self.counting = counting[fields_to_use]
        print(self.counting.columns)
        self.counting['date_time_0'] = self.counting.apply(
            lambda x: pd.Timestamp(year=x['Year'], month=x['Month'], day=x['Day'],
                                   hour=int(x['Time From'].split(':')[0]), minute=int(x['Time From'].split(':')[1])),
            axis=1)
        self.counting['date_time_1'] = self.counting.apply(
            lambda x: pd.Timestamp(year=x['Year'], month=x['Month'], day=x['Day'],
                                   hour=int(x['Time From'].split(':')[0]), minute=int(x['Time To'].split(':')[1])),
            axis=1)
        h=0


