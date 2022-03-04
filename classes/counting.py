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
    type_to_remove = ['C', 'D', 'NT', 'S', 'T', 'U']

    @staticmethod
    @abc.abstractmethod
    def column_names():
        pass

    @staticmethod
    def general_tasks(file: GeoDataFrame, counts_nodes: GeoDataFrame,
                      clip_file: GeoDataFrame) -> GeoDataFrame:
        """
        :param file: delete nan geometry and than reprojected, clip and do it unidirectional
        :param counts_nodes: for Azimuth calculation
        :param clip_file:
        :return: gdf
        """

        def __calculate_azimuth(p1, p2):
            try:
                return degrees(atan2(p2['XCOORD'] - p1['XCOORD'], p2['YCOORD'] - p1['YCOORD'])) % 360
            except:
                print('can not calculate azimuth')
                return -1

        def __remove_type(value):
            """
            This method remove segments appear in the list type_to_remove
            :param value:
            :return:
            """
            if value == 'R,U':
                return False
            if value is None:
                return True
            row_split = value.split(',')
            for val in row_split:
                if val not in Counting.type_to_remove:
                    return True
            return False

        print("_gis_oper_for_all")
        print('__clip')
        file = file[~file.is_empty]
        # "Sjoin" is used to remove the entire segment that intersects the clip.

        clipped = file.to_crs(Counting.crs).sjoin(clip_file, how="inner", predicate='within')
        clipped = clipped[~clipped.is_empty][file.columns]
        # calculate azimuth
        print("__calculate azimuth")
        counts_nodes.set_index('NO', inplace=True)
        clipped['azimuth'] = clipped.apply(
            lambda row: __calculate_azimuth(counts_nodes.loc[row['FROMNODENO']], counts_nodes.loc[row['TONODENO']]),
            axis=1)

        print("__calculate is_type")
        clipped['is_type'] = clipped['TSYSSET'].apply(__remove_type)
        return clipped[clipped['is_type']]

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


class CycleCount(Counting):

    @staticmethod
    def column_names():
        return ['walcycdata_id', 'cyclecount_id',
                'cyclecount_count', 'walcycdata_is_valid',
                'walcycdata_last_modified',
                'cyclecount_timestamp_start', 'cyclecount_timestamp_end', 'geometry']

    @staticmethod
    def cycle_count_def(count, counts_nodes: GeoDataFrame, clip_file: GeoDataFrame) -> GeoDataFrame:
        """
        This method calcultes
        :param count:
        :param counts_nodes: for Azimuth calculation
        :param clip_file:
        :return:
        """
        # project and clip
        count = Counting.general_tasks(file=count, counts_nodes=counts_nodes, clip_file=clip_file)
        # change table names
        count.rename(
            columns={"NO": "cyclecount_id", "ZAEHLUN~19": "cyclecount_count"}, inplace=True)

        # drop fields
        count = count[['FROMNODENO', 'TONODENO', 'cyclecount_id', 'cyclecount_count', 'TSYSSET', 'azimuth', 'geometry']]

        # new fields, walcycdata_id = a unique number with leading 2 for cycle count
        count.reset_index(inplace=True)
        n = len(count)
        count['walcycdata_id'] = pd.Series(map(lambda x: int('2' + x), np.arange(n).astype(str)))
        count['walcycdata_is_valid'], count['walcycdata_last_modified'], \
        count['osm_walcycdata_id'], count['cyclecount_timestamp_start'], count[
            'cyclecount_timestamp_end'] = 1, CycleCount.date, -1, -1, -1
        count.set_index('walcycdata_id', inplace=True)
        return count.drop('index', axis=1)


class CarCount(Counting):
    # ToDo should be combine with bike_def. in addation the function should be updated
    @staticmethod
    def car_count_def(count, counts_nodes: GeoDataFrame, clip_file: GeoDataFrame) -> GeoDataFrame:
        # project and clip
        count = Counting.general_tasks(file=count, counts_nodes=counts_nodes, clip_file=clip_file)
        # change table names
        count.rename(
            columns={"NO": "carcount_id", "ZOhlung_~7": "carcount_count"}, inplace=True)

        # drop fields
        count = count[['FROMNODENO', 'TONODENO', 'carcount_id', 'carcount_count', 'TSYSSET', 'azimuth', 'geometry']]

        # new fields, walcycdata_id = a unique number with leading 3 for cycle count
        count.reset_index(inplace=True)
        n = len(count)
        count['walcycdata_id'], count['walcycdata_is_valid'], count['walcycdata_last_modified'], \
        count['osm_walcycdata_id'], count['carcount_timestamp_start'], count[
            'carcount_timestamp_end'] = pd.Series(
            map(lambda x: int('3' + x), np.arange(n).astype(str))), 1, CarCount.date, -1, -1, -1
        count.set_index('walcycdata_id', inplace=True)
        return count.drop('index', axis=1)

    @staticmethod
    def column_names():
        return ['walcycdata_id', 'carcount_id',
                'carcount_count', 'walcycdata_is_valid',
                'walcycdata_last_modified',
                'carcount_timestamp_start', 'carcount_timestamp_end', 'geometry']


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
