from geopandas import GeoDataFrame
import geopandas as gpd
import osmnx as ox
from osmnx import downloader
from shapely.geometry import LineString
import pandas as pd
import numpy as np
from math import degrees, atan2
from classes.munich import *


class OSM(MunichData):

    date = pd.to_datetime("today")
    osm_column_names = ['walcycdata_id', 'walcycdata_is_valid', 'walcycdata_last_modified', 'osm_id', 'osm_fclass',
                        'osm_name', 'osm_oneway', 'osm_maxspeed', 'osm_layer', 'osm_bridge', 'osm_tunnel', 'geometry']

    @staticmethod
    def prepare_osm_data():
        print('prepare_osm_data')
        # define the tags that should be downloaded with the network
        useful_tags_path = ['osmid', 'highway', 'name', 'oneway', 'maxspeed', 'layer', 'bridge', 'tunnel']
        ox.utils.config(useful_tags_way=useful_tags_path)
        polygon = gpd.read_file(r'shp_files\munich_4326_large.shp')['geometry'][0]

        # Download data from osm interface based on polygon boundaries
        print('_download data')
        response_jsons = downloader._osm_network_download(polygon, 'all', None)
        elements = response_jsons[0]['elements']

        # Based on the downloaded nodes and ways, build the network
        nodes_list = {}
        ways = {}
        print('_build network')
        useful_tags = useful_tags_path[1:]
        for element in elements:
            if element["type"] == "node":
                nodes_list[element['id']] = (element['lon'], element['lat'])
            elif element["type"] == "way":
                ways[element['id']] = dict([tag for tag in element['tags'].items() if tag[0] in useful_tags])
                ways[element['id']]['geometry'] = LineString([nodes_list[id_loc] for id_loc in element['nodes']])

        # Convert the ways dictionary into a dataframe
        edges_shp = GeoDataFrame(ways.values(), index=ways.keys(), crs="EPSG:4326")
        edges_shp['osmid'] = ways.keys()
        print("finish prepare_osm_data ")
        return edges_shp.to_crs(OSM.crs)

    @staticmethod
    def osm_file_def(osm_df: GeoDataFrame, cat_file: pd.DataFrame):
        # drop duplicate
        osm_df.drop_duplicates(subset=['osmid'], inplace=True)
        # change table names
        osm_df.rename(
            columns={"osmid": "osm_id", "name": "osm_name", "oneway": "osm_oneway",
                     "maxspeed": "osm_maxspeed",
                     "layer": "osm_layer", "bridge": "osm_bridge", "tunnel": "osm_tunnel"}, inplace=True)
        # categorical values  - f_class

        osm_df = osm_df[osm_df["highway"].isin(cat_file['categories'])]
        cat_file.set_index('categories', inplace=True)
        osm_df["osm_fclass"] = cat_file.loc[osm_df["highway"]]['values'].values
        osm_df["osm_fclass_hir"] = cat_file.loc[osm_df["highway"]]['hierarchical values'].values

        # bool values  -bridge,tunnel
        for field in ['osm_bridge', 'osm_tunnel', "osm_oneway"]:
            osm_df[field].loc[~osm_df[field].isnull()] = 1  # not nan
            osm_df[field].loc[(osm_df[field].isnull()) | (osm_df[field] == 'no')] = 0
        osm_df['osm_name'].fillna('no_name', inplace=True)

        # new fields, walcycdata_id = a unique number with leading 4 for osm
        osm_df.reset_index(inplace=True)
        n = len(osm_df)
        osm_df['walcycdata_id'], osm_df['walcycdata_is_valid'], osm_df['walcycdata_last_modified'] = pd.Series(
            map(lambda x: int('4' + x), np.arange(
                n).astype(str))), 1, OSM.date

        # more changes should be aplie
        to_int8 = ['walcycdata_is_valid', 'osm_bridge', 'osm_tunnel', "osm_oneway"]
        osm_df[to_int8] = osm_df[to_int8].astype(np.int8)

        to_int = ['osm_maxspeed', "osm_layer"]
        osm_df[to_int] = osm_df[to_int].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).astype(int)

        # Obtain azimuth
        def azimuth_osm(geometry):
            pnt_0 = geometry.coords[0]
            pnt_1 = geometry.coords[-1]
            if pnt_0 == pnt_1:
                return -1
            else:
                return degrees(atan2(pnt_1[0] - pnt_0[0], pnt_1[1] - pnt_0[1])) % 360

        osm_df['azimuth'] = osm_df['geometry'].apply(azimuth_osm)
        return osm_df.drop('index', axis=1)

    @staticmethod
    def find_the_opposite_roads(osm_gdf: GeoDataFrame):
        """

        :param osm_gdf:
        :return:
        """
        print('_find_the_opposite_roads')
        one_way = osm_data[osm_gdf['osm_oneway'] == 1]
        one_ways = one_way.groupby('highway').apply(OSM.__find_optimal_applicant)
        one_ways.reset_index(level=['highway'], inplace=True)
        osm_data.set_index('osm_id', inplace=True)
        osm_data['pair'] = one_ways['osm_id_2']
        print('_write to the disk')
        return osm_data

    @staticmethod
    def __find_optimal_applicant(group):
        print('___' + group.name)
        group_buffer = GeoDataFrame(group[['highway', 'azimuth', 'osm_id']],
                                    geometry=group.geometry.buffer(25, cap_style=2), crs=OSM.crs)
        group_buffer2 = GeoDataFrame(group[['highway', 'azimuth', 'osm_id']],
                                     geometry=group.geometry.buffer(25, cap_style=2), crs=OSM.crs)
        overlay = group_buffer.overlay(group_buffer2, how='intersection')
        overlay = overlay[overlay['osm_id_1'] != overlay['osm_id_2']]
        overlay['area'] = overlay.area
        results = overlay.groupby('osm_id_1').apply(OSM.__select_from_options)
        results = results[results['osm_id_1'].notna()].set_crs(overlay.crs).drop(columns=['osm_id_1'])
        return results

    @staticmethod
    def __select_from_options(group):
        if len(group) > 1:
            group.sort_values('area', ascending=False, inplace=True)
            for row_temp in group.iterrows():
                row = row_temp[1]
                if abs((abs(row['azimuth_1'] - row['azimuth_2']) - 180)) < 10:
                    return row
        else:
            row = group.iloc[0]
            if abs((abs(row['azimuth_1'] - row['azimuth_2']) - 180)) < 10:
                return row


if __name__ == '__main__':
    import os

    os.chdir(r'D:\Users\Technion\Sagi Dalyot - AchituvCohen\WalCycData\shared_project')
    OSM.prepare_osm_data()
    osm_data = gpd.read_file('shp_files/matching_files.gpkg', layer='osm_gdf')
    OSM.find_the_opposite_roads(osm_gdf=osm_data).to_file("shp_files/two_ways.gpkg",
                                                          layer='osm_gdf', driver="GPKG")
