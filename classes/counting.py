from abc import ABC

from geopandas import GeoDataFrame
import geopandas as gpd
import osmnx as ox
from osmnx import downloader
from shapely.geometry import LineString, Point
import pandas as pd
import numpy as np
from math import degrees, atan2
from classes.munich import *
import pickle
import networkx as nx
import abc


class Counting(MunichData, metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def column_names():
        pass

    @staticmethod
    def update_matching(my_combine_network: GeoDataFrame, counting_df: GeoDataFrame, user_type: str) -> GeoDataFrame:
        # Define index
        index_name = 'walcycdata_id'
        osm_name = 'osm_walcycdata_id'
        my_combine_network = my_combine_network[my_combine_network[index_name + '_' + user_type].notna()]
        my_combine_network.index = my_combine_network.index.astype('int')
        my_combine_network.set_index(index_name + '_' + user_type, inplace=True)
        counting_df.set_index(index_name, inplace=True)
        # Update date about counting
        counting_df[osm_name] = my_combine_network[osm_name]
        return counting_df


class CarCount(Counting):

    @staticmethod
    def column_names():
        return ['walcycdata_id', 'carcount_id',
                'carcount_count', 'walcycdata_is_valid',
                'walcycdata_last_modified', 'osm_id',
                'carcount_timestamp_start', 'carcount_timestamp_end', 'geometry']


class CycleCount(Counting):

    @staticmethod
    def column_names():
        return ['walcycdata_id',  'cyclecount_id',
                'cyclecount_count', 'walcycdata_is_valid',
                'walcycdata_last_modified', 'osm_id',
                'cyclecount_timestamp_start', 'cyclecount_timestamp_end', 'geometry']


if __name__ == '__main__':
    import os

    os.chdir(r'D:\Users\Technion\Sagi Dalyot - AchituvCohen\WalCycData\shared_project')

    my_cycle_count = gpd.read_file("shp_files/pr_data.gpkg", layer='cycle_count', driver="GPKG")
    my_car_count = gpd.read_file("shp_files/pr_data.gpkg", layer='car_count', driver="GPKG")
    combine_network = gpd.read_file("shp_files/matching_files.gpkg", layer='refine_matching', driver="GPKG")
    Counting.update_matching(my_combine_network=combine_network, counting_df=my_car_count, user_type='car').to_file(
        "shp_files/pr_data.gpkg", layer='car_count2', driver="GPKG")
    Counting.update_matching(my_combine_network=combine_network, counting_df=my_cycle_count, user_type='cycle').to_file(
        "shp_files/pr_data.gpkg", layer='cycle_count2', driver="GPKG")
