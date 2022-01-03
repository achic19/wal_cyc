from geopandas import GeoDataFrame
import geopandas as gpd
import osmnx as ox
from osmnx import downloader
from shapely.geometry import LineString, Point
import pandas as pd
from pandas import DataFrame
import numpy as np
from math import degrees, atan2, sqrt
from classes.munich import *
import pickle
import networkx as nx


class OSM(MunichData):
    # Data

    osm_column_names = ['walcycdata_id', 'walcycdata_is_valid', 'walcycdata_last_modified', 'osm_id', 'osm_fclass',
                        'osm_name', 'osm_oneway', 'osm_maxspeed', 'osm_layer', 'osm_bridge', 'osm_tunnel',
                        'cyclecount_count', 'carcount_count', 'num_incidents', 'geometry']

    @staticmethod
    def prepare_osm_data(polygon):
        print('_prepare_osm_data')
        # define the tags that should be downloaded with the network
        useful_tags_path = ['osmid', 'highway', 'name', 'oneway', 'maxspeed', 'layer', 'bridge', 'tunnel']
        ox.utils.config(useful_tags_way=useful_tags_path)
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
                ways[element['id']]['from'] = element['nodes'][0]
                ways[element['id']]['to'] = element['nodes'][-1]

        # Convert the ways dictionary into a dataframe
        print('_create and save graph ')
        edges_shp = GeoDataFrame(ways.values(), index=ways.keys(), crs="EPSG:4326")
        edges_shp['osmid'] = ways.keys()

        print("finish prepare_osm_data ")
        osm_as_graph = OSMAsObject(nodes_list, edges_shp)

        return edges_shp.to_crs(OSM.crs), osm_as_graph.gdp_pnt

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

        # more changes should be applied
        to_int8 = ['walcycdata_is_valid', 'osm_bridge', 'osm_tunnel', "osm_oneway"]
        osm_df[to_int8] = osm_df[to_int8].astype(np.int8)

        to_int = ['osm_maxspeed', "osm_layer"]
        osm_df[to_int] = osm_df[to_int].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).astype(int)

        # Obtain azimuth
        def __azimuth_osm(geometry):
            pnt_0 = geometry.coords[0]
            pnt_1 = geometry.coords[-1]
            if pnt_0 == pnt_1:
                return -1
            else:
                return degrees(atan2(pnt_1[0] - pnt_0[0], pnt_1[1] - pnt_0[1])) % 360

        osm_df['azimuth'] = osm_df['geometry'].apply(__azimuth_osm)
        return osm_df.drop('index', axis=1)

    @staticmethod
    def statistic(two_ways_db: GeoDataFrame):

        def __stat_by_highway(group):
            print(group.name)
            len_highway = len(group)
            not_found = len(group[group['pair'] == str(-1)])
            not_found_percentage = not_found / len_highway * 100
            found = len_highway - not_found
            found_percentage = found / len_highway * 100
            rep[group.name] = [len_highway, found, found_percentage, not_found, not_found_percentage]

            # calculate the number of one-ways roads

        one_way = two_ways_db[two_ways_db['osm_oneway'] == 1]
        len_two_ways = len(two_ways_db)
        len_one_way = len(one_way)
        print("There ara {} one-ways roads which is {}%".format(len_one_way, len_one_way / len_two_ways * 100))

        # Dictionary to solve the results
        rep = {}
        one_way.groupby('highway').apply(__stat_by_highway)
        res = pd.DataFrame(index=rep.keys(), data=rep.values(),
                           columns=['count', 'found_count', 'found_count_percentage', 'not found_count',
                                    'not found_count_percentage'])
        print(res)
        return res

    @staticmethod
    def from_local_to_osm(osm_network: GeoDataFrame, local_matching_network: GeoDataFrame) -> GeoDataFrame:
        """
        This method update osm network with car and bike countings
        :param osm_network:
        :param local_matching_network:
        :return:
        """
        print('_matching_to_osm_counting')
        print('__preparation')
        # make sure osm_id is the index
        osm_network.set_index('osm_id', inplace=True)
        # form str  to int list in local_matching_network
        local_matching_network['osm_ids'] = local_matching_network['osm_ids'].apply(
            lambda x: [int(i) for i in x.split(',')])
        # null to zero
        local_matching_network['carcount_count'] = local_matching_network['carcount_count'].fillna(0)
        #  Use only rows from local_matching_network with counting and with matching OSM,
        local_matching_network = local_matching_network[
            (local_matching_network['carcount_count'] > 0) & (local_matching_network['cyclecount_count'] > 0) & (
                    local_matching_network['osm_walcycdata_id'] != -1)]
        # Add four new two columns sum and time
        avg_count_columns_names = ['list_cycle_count', 'list_cycle_length', 'list_car_count', 'list_car_length']
        avg_count_columns_names_opp = ['list_cycle_count_opp', 'list_cycle_length_opp', 'list_car_count_opp',
                                       'list_car_length_opp']

        osm_network[avg_count_columns_names] = [[[] for _ in range(len(avg_count_columns_names))] for _ in
                                                range(len(osm_network))]
        osm_network[avg_count_columns_names_opp] = [[[] for _ in range(len(avg_count_columns_names_opp))] for _ in
                                                    range(len(osm_network))]
        local_matching_network['length'] = local_matching_network.length

        def calculate_counting(row):
            # The method add count information based on the date in row

            def find_direction_update(osm_id):
                """
                This method updates the proper columns by azimuths of osm and local objects
                 (same direction or opposite one)
                :param osm_id:
                :return:
                """

                def count_per_row(columns_to_update: list):
                    """
                    Update count for the given osm_id
                    :param columns_to_update: list of columns names to updata
                    :return:
                    """
                    if row['cyclecount_count'] > 0:
                        osm_network.at[osm_id, columns_to_update[0]].append(row['cyclecount_count'])
                        osm_network.at[osm_id, columns_to_update[1]].append(row.length)
                    if row['carcount_count'] > 0:
                        osm_network.at[osm_id, columns_to_update[2]].append(row['carcount_count'])
                        osm_network.at[osm_id, columns_to_update[3]].append(row.length)

                osm_azimuth = osm_network.at[osm_id, 'azimuth']
                is_opposite_angle = abs(row.azimuth - osm_azimuth)
                if 30 < is_opposite_angle < 330:
                    count_per_row(avg_count_columns_names_opp)
                else:
                    count_per_row(avg_count_columns_names)

            # take all the values from start_osm_id and end_osm_id and remove osm_walcycdata_id and for all
            [find_direction_update(item) for item in row['osm_ids']]

        # loop over each filtered matching network:
        print('__calculate_counting for osm')
        local_matching_network.apply(calculate_counting, axis=1)

        # calculate average

        def weighted_avg_and_std(values, weights):
            """
            Return the weighted average and standard deviation.

            values, weights -- Numpy ndarrays with the same shape.
            """
            if len(values) == 0:
                return 0, 0
            if len(values) == 1:
                return values[0], 0
            average = np.average(values, weights=weights)
            # Fast and numerically precise:
            variance = np.average((values - average) ** 2, weights=weights)
            return average, sqrt(variance)

        print('__calculate average_std')

        osm_network[['cycle_count', 'cycle_count_std']] = list(osm_network.apply(
            lambda x: weighted_avg_and_std(x['list_cycle_count'], x['list_cycle_length']), axis=1))

        osm_network[['cycle_count_opp', 'cycle_count_std_opp']] = list(osm_network.apply(
            lambda x: weighted_avg_and_std(x['list_cycle_count_opp'], x['list_cycle_length_opp']), axis=1))

        osm_network[['car_count', 'car_count_std']] = list(osm_network.apply(
            lambda x: weighted_avg_and_std(x['list_car_count'], x['list_car_length']), axis=1))

        osm_network[['car_count_opp', 'car_count_std_opp']] = list(osm_network.apply(
            lambda x: weighted_avg_and_std(x['list_car_count_opp'], x['list_car_length_opp']), axis=1))

        print('__drop processing fields')
        avg_count_columns_names.extend(avg_count_columns_names_opp)
        osm_network.drop(columns=avg_count_columns_names, inplace=True)
        return osm_network

    @staticmethod
    def from_incident_to_osm(osm_network: GeoDataFrame, local_incidents: GeoDataFrame) -> GeoDataFrame:
        """
        This method update osm network with car and bike countings
        :param osm_network:
        :param local_incidents:
        :return:
        """
        print('_from_incident_to_osm')
        # make sure osm_id is the index
        osm_network.set_index('osm_id', inplace=True)
        osm_network['num_incidents'] = local_incidents.groupby('osm_walcycdata_id')['index'].count()
        osm_network['num_incidents'] = osm_network['num_incidents'].fillna(0)
        return osm_network

    @staticmethod
    def find_the_opposite_roads(osm_gdf: GeoDataFrame):
        """

        :param osm_gdf:
        :return:
        """
        print('_find_the_opposite_roads')
        one_way = osm_gdf[osm_gdf['osm_oneway'] == 1]
        one_ways = one_way.groupby('highway').apply(OSM.__find_optimal_applicant)
        one_ways.reset_index(level=['highway'], inplace=True)
        osm_gdf.set_index('osm_id', inplace=True)
        osm_gdf['pair'] = one_ways[0]
        osm_gdf['pair'] = osm_gdf['pair'].fillna(-1)
        osm_gdf['pair'] = osm_gdf['pair'].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x))
        osm_gdf['pair'] = osm_gdf['pair'].replace('', '-1')
        print('_write to the disk')
        return osm_gdf

    @staticmethod
    def __find_optimal_applicant(group):
        print('___' + group.name)
        group_buffer = GeoDataFrame(group[['highway', 'azimuth', 'osm_id']],
                                    geometry=group.geometry.buffer(30, cap_style=2), crs=OSM.crs)
        group_buffer2 = GeoDataFrame(group[['highway', 'azimuth', 'osm_id']],
                                     geometry=group.geometry.buffer(30, cap_style=2), crs=OSM.crs)
        overlay = group_buffer.overlay(group_buffer2, how='intersection')
        overlay = overlay[overlay['osm_id_1'] != overlay['osm_id_2']]
        overlay['area'] = overlay.area
        results = overlay.groupby('osm_id_1').apply(OSM.__select_from_options)
        return results

    @staticmethod
    def __select_from_options(group):
        if len(group) > 1:
            results = []
            for row_temp in group.iterrows():
                row = row_temp[1]
                if abs((abs(row['azimuth_1'] - row['azimuth_2']) - 180)) < 35 and row['area'] > 30:
                    results.append(row['osm_id_2'])
            return results
        else:
            row = group.iloc[0]
            if abs((abs(row['azimuth_1'] - row['azimuth_2']) - 180)) < 35 and row['area'] > 30:
                return row['osm_id_2']
            else:
                return []


class OSMAsObject(MunichData):

    def __init__(self, nodes_list, edges_shp):
        print("__gdp_pnt ")
        nodes_coor = [Point(f[0], f[1]) for f in nodes_list.values()]
        self.gdp_pnt = gpd.GeoDataFrame(nodes_list.keys(), geometry=nodes_coor, crs="EPSG:4326").to_crs(OSM.crs)
        self.gdp_pnt.rename(columns={0: 'id'}, inplace=True)

        # save the osm data as graph
        self.graph = nx.from_pandas_edgelist(edges_shp, 'from', 'to', edge_attr='osmid')
        with open('OSMAsOsm.pkl', 'wb') as osm_as_graph:
            print('__write_to_disk_with_pickle')
            pickle.dump(self, osm_as_graph)

    @staticmethod
    def create_osm_obj_from_local_machine():
        print('_load_osm_as_object')
        with open('OSMAsOsm.pkl', 'rb') as osm_as_object:
            return pickle.load(osm_as_object)


if __name__ == '__main__':
    import os

    os.chdir(r'D:\Users\Technion\Sagi Dalyot - AchituvCohen\WalCycData\shared_project')
    refine_matching = gpd.read_file("shp_files/matching_files.gpkg", layer='refine_matching', driver="GPKG")
    my_osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network')
    res = OSM.from_local_to_osm(my_osm_data, refine_matching)
    res.to_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network2')
