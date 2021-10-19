import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import networkx as nx
import osmnx as ox
from geopandas import GeoDataFrame
from shapely.coords import CoordinateSequence
import math


def list_to_str(row, columns):
    """
    convert list in columns to string
    :param row:
    :param columns:
    :return:
    """
    try:
        for column in columns:
            if isinstance(row[column], list):
                row[column] = row[column][0]
    except:
        pass
    return row


def prepare_osm_data():
    print('prepare_osm_data')
    # define the tags that should be downloaded with the network
    useful_tags_path = ['osmid', 'highway', 'name', 'oneway', 'maxspeed', 'layer', 'bridge', 'tunnel']
    ox.utils.config(useful_tags_way=useful_tags_path)
    # download data from OSM based on polygon boundaries
    area = gpd.read_file(r'shp_files\munich_4326_large.shp')['geometry'][0]
    graph = ox.graph_from_polygon(area, network_type='all', clean_periphery=False)
    # project to espg:3857 ,create undirected and create gdf
    graph_pr = ox.project_graph(graph)
    graph_pr = graph_pr.to_undirected()
    gdf_format = ox.graph_to_gdfs(graph_pr)
    edges = gdf_format[1]

    columns = [value for value in useful_tags_path if value in list(edges.columns)]

    edges_new = edges.apply(lambda row: list_to_str(row, columns), axis=1)
    edges_new.crs = edges.crs
    columns.append('geometry')
    edges_shp = edges_new.loc[:, columns]
    print('to shape_file')
    edges_shp.to_crs(epsg=3857).to_file(r'shp_files\osm_road_network.shp')


def osm_file_def():
    osm_df = gpd.read_file('shp_files/osm_road_network.shp')

    # drop duplicate
    osm_df.drop_duplicates(subset=['geometry'], inplace=True)
    # change table names
    osm_df.rename(
        columns={"osmid": "osm_id", "name": "osm_name", "oneway": "osm_oneway",
                 "maxspeed": "osm_maxspeed",
                 "layer": "osm_layer", "bridge": "osm_bridge", "tunnel": "osm_tunnel"}, inplace=True)
    # categorical values  - f_class
    cat_file = pd.read_csv('shp_files/cat.csv', header=None, names=['values', 'categories'])
    osm_df = osm_df[osm_df["highway"].isin(cat_file['categories'])]
    cat_file.set_index('categories', inplace=True)
    osm_df["osm_fclass"] = cat_file.loc[osm_df["highway"]]['values'].values

    # null in layer = ground level = 0

    osm_df["osm_layer"].fillna(0, inplace=True)
    # bool values  -bridge,tunnel
    osm_df = to_bool(osm_df, 'osm_bridge')
    osm_df = to_bool(osm_df, 'osm_tunnel')

    # drop fields

    # new fields, walcycdata_id = a unique number with leading 4 for osm
    osm_df.reset_index(inplace=True)
    n = len(osm_df)
    osm_df['walcycdata_id'], osm_df['walcycdata_is_valid'], osm_df['walcycdata_last_modified'] = pd.Series(
        map(lambda x: int('4' + x), np.arange(
            n).astype(str))), True, date
    osm_df.drop(columns=['index', 'u', 'v', 'key'], inplace=True)
    return osm_df


def to_bool(osm_df, field):
    osm_df[field].loc[~osm_df[field].isnull()] = 1  # not nan
    osm_df[field].loc[(osm_df[field].isnull()) | (osm_df[field] == 'no')] = 0  # not nan
    return osm_df


def clear_footway_near_other_elements(osm_network: GeoDataFrame):
    """
    When the footway is around other elements, remove them for later use based on two conditions: 1. they are at the
    same level and they quite parallel
    :return:
    """
    footway = osm_network.loc['footway'].set_index('walcycdata_id')
    roads = osm_network.drop(labels=['footway', 'path', 'service'], axis=0).set_index('walcycdata_id')
    buffer = gpd.GeoDataFrame(footway[['osm_bridge', 'osm_tunnel']], crs=osm_network.crs,
                              geometry=footway.geometry.buffer(20, cap_style=2))
    # For evaluation only
    buffer.to_file("shp_files/pr_data.gpkg", layer='buffer_footway', driver="GPKG")
    footway_with_roads = buffer.reset_index().sjoin(roads, how='inner', predicate='crosses')
    footway_with_roads_contains = buffer.reset_index().sjoin(roads, how='inner', predicate='contains')
    footway_with_roads = footway_with_roads.append(footway_with_roads_contains)
    footway_with_roads_same_level = footway_with_roads[
        (footway_with_roads['osm_tunnel_left'] == footway_with_roads['osm_tunnel_right']) & (
                footway_with_roads['osm_bridge_left'] == footway_with_roads['osm_bridge_right'])]
    footway_with_roads_same_level['is_parallel'] = footway_with_roads_same_level.apply(
        lambda x: is_parallel(footway.loc[x['walcycdata_id']].geometry.coords,
                              roads.loc[x['index_right']].geometry.coords
                              ), axis=1)
    footway_with_roads_same_level = footway_with_roads_same_level[footway_with_roads_same_level['is_parallel']]
    return osm_network[~osm_network['walcycdata_id'].isin(footway_with_roads_same_level['walcycdata_id'].unique())]


def is_parallel(geo_1: CoordinateSequence, geo_2: CoordinateSequence):
    """
    This function check whether two geometries are parallel (with a tolerance of 20 degrees)
    :param geo_1:
    :param geo_2:
    :return:
    """
    azimuth_list = []
    for geo in [geo_1, geo_2]:
        azimuth = math.degrees(math.atan2(geo[-1][0] - geo[0][0], geo[-1][1] - geo[0][1]))
        if azimuth < 0:
            azimuth += 360
        azimuth_list.append(azimuth)
    if (abs(azimuth_list[1] - azimuth_list[0]) < 15) or (abs((abs(azimuth_list[1] - azimuth_list[0]) - 180)) < 15):
        return True
    else:
        return False


def incident_def():
    # upload files
    incident = gpd.read_file('shp_files/Incident_data.shp')

    # change table names
    incident.rename(
        columns={"OBJECTID": "incident_id", "Land": "Incident_land", "Administ_1": "Incident_region",
                 "Administra": "Incident_district", "Year_of_ac": "Incident_year", "Month_of_a": "Incident_month",
                 "Day_of_wee": "Incident_week_day", "Hour_of_ac": "Incident_hour",
                 "Pernosal_i": "Incident_personal_injury", "Kinds_of_a": "Incident_class",
                 "Type_of_ac": "Incident_type", "Light_cond": "Incident_light_condition",
                 "Bicycle": "Incident_with_bycycle", "Passenger_": "Incident_with_passenger_car",
                 "Passenger": "Incident_with_passenger", "Motorcycle": "Incident_with_motorcyle",
                 "Goods_vehi": "Incident_with_goods", "Other": "Incident_with_other",
                 "Surface_co": "Incident_surface_condition", "LINREFX": "Incident_ref_x", "LINREFY": "Incident_ref_y",
                 "XGCSWGS84": "Incident_gcswgs84_x", "YGCSWGS84": "Incident_gcswgs84_y"}, inplace=True)
    incident[
        ['Incident_with_bycycle', 'Incident_with_passenger_car', 'Incident_with_passenger', 'Incident_with_motorcyle',
         'Incident_with_goods', 'Incident_with_other']] = incident[
        ['Incident_with_bycycle', 'Incident_with_passenger_car', 'Incident_with_passenger', 'Incident_with_motorcyle',
         'Incident_with_goods', 'Incident_with_other']].astype('bool')

    # drop fields
    incident.drop(
        columns=['Municipali', 'severity'], inplace=True)

    # new fields, , walcycdata_id = a unique number with leading 1 for incidents
    incident.reset_index(inplace=True)
    n = len(incident)
    incident['walcycdata_id'], incident['walcycdata_is_valid'], incident['walcycdata_last_modified'], incident[
        'osm_walcycdata_id'] = pd.Series(map(lambda x: int('1' + x), np.arange(n).astype(str))), True, date, ''
    incident = general_tasks(file=incident, unidirectional=False)
    return incident


def join_to_bike_network(incident_gdb: GeoDataFrame, network: GeoDataFrame) -> GeoDataFrame:
    """
    Find for each incident the closet street segment (with bike 'NO')
    :param incident_gdb:
    :param network:
    :return:
    """

    # remove from the network rows with Null values in 'Cyclecount_id'
    network = GeoDataFrame(network[network['Cyclecount_id'].notna()]['Cyclecount_id'], geometry=network.geometry,
                           crs=network.crs)
    return gpd.sjoin_nearest(incident_gdb, network, how='left', distance_col='dist')


def cycle_count_def(count):
    # project and clip
    count = general_tasks(file=count, count_column="ZAEHLUN~19", unidirectional=True)
    # change table names
    count.rename(
        columns={"NO": "Cyclecount_id", "count": "Cyclecount_count"}, inplace=True)

    # drop fields
    count = count[['Cyclecount_id', 'Cyclecount_count', 'geometry']]

    # new fields, walcycdata_id = a unique number with leading 2 for cycle count
    count.reset_index(inplace=True)
    n = len(count)
    count['walcycdata_id'] = pd.Series(map(lambda x: int('2' + x), np.arange(n).astype(str)))
    count['walcycdata_is_valid'], count['walcycdata_last_modified'], \
    count['osm_walcycdata_id'], count['Cyclecount_timestamp_start'], count[
        'Cyclecount_timestamp_end'] = True, date, '', '', ''
    return count


def car_count_def(count):
    # project and clip
    count = general_tasks(file=count, count_column="ZOhlung_~7", unidirectional=True)
    # change table names
    count.rename(
        columns={"NO": "Carcount_id", "count": "Carcount_count"}, inplace=True)

    # drop fields
    count = count[['Carcount_id', 'Carcount_count', 'geometry']]

    # new fields, walcycdata_id = a unique number with leading 3 for cycle count
    count.reset_index(inplace=True)
    n = len(count)
    count['walcycdata_id'], count['walcycdata_is_valid'], count['walcycdata_last_modified'], \
    count['osm_walcycdata_id'], count['Carcount_timestamp_start'], count[
        'Carcount_timestamp_end'] = pd.Series(
        map(lambda x: int('3' + x), np.arange(n).astype(str))), True, date, '', '', ''
    return count


def general_tasks(file: GeoDataFrame, unidirectional: bool, count_column: str = '') -> GeoDataFrame:
    """
    :param unidirectional: The code performs unidirectional operations if True
    :param count_column: group_by this column
    :param file: delete nan geometry and than reprojected, clip and do it unidirectional
    :return: gdf
    """
    print("gis_oper_for_all")
    file = file[~file.is_empty]
    clipped = gpd.clip(file.to_crs(crs), clip_file)
    clipped = clipped[~clipped.is_empty]
    if unidirectional:
        # make it "unidirectional "grouping by "NO'(sum) and than drop_duplicates 'NO'
        clipped.fillna(0, inplace=True)
        group_by = clipped.groupby(['NO']).sum()
        clipped.drop_duplicates(subset=['NO'], inplace=True)
        clipped.set_index('NO', inplace=True)
        clipped['count'] = group_by[count_column]
        return clipped.reset_index()
    else:
        return clipped


def count_osm(osm_gdf: GeoDataFrame, is_cycle: int, name_id):
    """
    This function matches the count data with corresponding OSM entities.
    :param osm_gdf: The OSM network to work on
    :param is_cycle: Based on this bool, we sort the relevant information from count data
    1:only bikes 2:only cars 3:bikes and cars
    :param name_id: This will allow us to  identify the id to whom the osm entity will be linked
    :return:
    """
    if is_cycle == 1:
        count_data_local = count_data[count_data['Carcount_count'] == 0]
    elif is_cycle == 2:
        count_data_local = count_data[count_data['Cyclecount_count'] == 0]
    else:
        count_data_local = count_data[(count_data['Cyclecount_count'] > 0) & (count_data['Carcount_count'] > 0)]

    network = GeoDataFrame(osm_gdf[['osm_id', 'walcycdata_id']],
                           geometry=osm_gdf.geometry.buffer(10, cap_style=2), crs=osm_gdf.crs)
    buffer_count = gpd.GeoDataFrame(count_data_local[[name_id]], crs=count_data_local.crs,
                                    geometry=count_data_local.geometry.buffer(10, cap_style=2))
    res_inter = buffer_count.overlay(network, how='intersection')


if __name__ == '__main__':
    # general code
    clip_file = gpd.read_file('shp_files/munich_3857.shp')
    crs = "EPSG:3857"
    date = pd.to_datetime("today")
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')

    params = {'osm': [False, {'prepare_osm_data': False, 'osm_file': False, 'car_bike_osm': True}],
              'count': [False,
                        {'cycle_count': False, 'car_count': False, 'merge_files': False, 'delete_null_zero': False}],
              'incident': [False, {'prepare_incident': False, 'join_to_bike_network': False}],
              'count_osm': [True, {'bikes': True, 'cars': False, 'bikes and car': False}],
              'data_to_server': False}

    if params['osm'][0]:
        local_params = params['osm'][1]
        if local_params['prepare_osm_data']:
            prepare_osm_data()
        if local_params['osm_file']:
            print('create osm table')
            my_osm_file = osm_file_def()
            my_osm_file.to_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network', driver="GPKG")
        # my_osm_file.to_postgis(name="openstreetmap_road_network", con=engine, schema='production',
        #                        if_exists='replace')
        if local_params['car_bike_osm']:
            print('delete superfluous information')
            osm_file = gpd.read_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network')
            bike_osm = osm_file.set_index('highway').drop(labels=['steps'], axis=0)
            bike_osm = clear_footway_near_other_elements(bike_osm)
            car_osm = bike_osm.drop(labels=['footway', 'cycleway', 'bridleway'], axis=0)
            bike_osm.to_file("shp_files/pr_data.gpkg", layer='cycle_osm', driver="GPKG")
            car_osm.reset_index().to_file("shp_files/pr_data.gpkg", layer='car_osm', driver="GPKG")
    if params['count'][0]:
        local_params = params['count'][1]

        if local_params['cycle_count']:
            print('create cycle_count table')
            # upload files
            cycle_count = gpd.read_file('shp_files/cycle_count_data.shp')
            my_cycle_count = cycle_count_def(cycle_count)
            my_cycle_count.to_file("shp_files/pr_data.gpkg", layer='cycle_count', driver="GPKG")

            # print('upload cycle_count table')
            # my_cycle_count.to_postgis(name="cycle_count_data", con=engine, schema='production',
            #                           if_exists='replace')

        if local_params['car_count']:
            print('create car_count table')
            car_count = gpd.read_file('shp_files/cars_count_data.shp')
            my_car_count = car_count_def(car_count)
            my_car_count.to_file("shp_files/pr_data.gpkg", layer='car_count', driver="GPKG")
            # my_car_count.to_postgis(name="car_count_data", con=engine, schema='production',
            #                         if_exists='replace')

        if local_params['merge_files']:
            cycle_count = gpd.read_file("shp_files/pr_data.gpkg", layer='cycle_count')
            car_count = gpd.read_file("shp_files/pr_data.gpkg", layer='car_count')
            all_count = cycle_count.merge(right=car_count, left_on='Cyclecount_id',
                                          right_on='Carcount_id', how='outer', suffixes=('_cycle', '_car'))
            # 'geometry_cycle' will be the geometry for all the the entities in the new file
            all_count.loc[all_count['geometry_cycle'].isnull(), 'geometry_cycle'] = \
                all_count[all_count['geometry_cycle'].isnull()]['geometry_car']
            all_count = GeoDataFrame(all_count, geometry=all_count['geometry_cycle'], crs=car_count.crs)

            all_count.drop(['geometry_car', 'geometry_cycle'], axis=1, inplace=True)
            all_count.to_file("shp_files/pr_data.gpkg", layer='all_count', driver="GPKG")
        if local_params['delete_null_zero']:
            all_count = gpd.read_file("shp_files/pr_data.gpkg", layer='all_count')
            all_count[['Cyclecount_count', 'Carcount_count']] = all_count[
                ['Cyclecount_count', 'Carcount_count']].fillna(0)
            all_count = all_count[(all_count['Cyclecount_count'] != 0) | (all_count['Carcount_count'] != 0)]
            all_count.to_file("shp_files/pr_data.gpkg", layer='all_count_no_zero', driver="GPKG")

    if params['incident'][0]:
        print('work on incident data')
        local_params = params['incident'][1]
        if local_params['prepare_incident']:
            my_incident = incident_def()
            my_incident.to_file("shp_files/pr_data.gpkg", layer='incident', driver="GPKG")
        if local_params['join_to_bike_network']:
            print('join_to_bike_network')
            my_incident = gpd.read_file("shp_files/pr_data.gpkg", layer='incident')
            all_count = gpd.read_file("shp_files/pr_data.gpkg", layer='all_count')
            my_incident = join_to_bike_network(my_incident, all_count)
            my_incident.to_file("shp_files/pr_data.gpkg", layer='incident_with_street_id', driver="GPKG")
        # my_incident.to_postgis(name="incident_data", con=engine, schema='production',
        #                        if_exists='replace')
    if params['count_osm'][0]:
        local_params = params['count_osm'][1]
        count_data = gpd.read_file("shp_files/pr_data.gpkg", layer='all_count_no_zero')

        if local_params['bikes']:
            print('join bikes count data to OSM network')
            osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='cycle_osm')
            count_osm(osm_data, 1, 'Cyclecount_id')
            # this includes only bikes users

            # bike_join.to_file("shp_files/pr_data.gpkg", layer='bike_join', driver="GPKG")

            #  To evaluate the results the code store
            # a seperate layer with entities found to be closed to count entities
            # bikes_osm = osm_data[osm_data['walcycdata_id'].isin(bike_join['walcycdata_id'].unique())]
            # bikes_osm.to_file("shp_files/pr_data.gpkg", layer='bikes_osm_join', driver="GPKG")

        if local_params['cars']:
            print('join  cars count data to OSM network')
            osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='car_osm')
            count_osm(osm_data, 2, 'Carcount_id')

            # cars_bike_join.to_file("shp_files/pr_data.gpkg", layer='cars_bike_join', driver="GPKG")

            #  To evaluate the results the code store
            # a seperate layer with entities found to be closed to count entities
            # cars_bikes_osm = osm_data[osm_data['walcycdata_id'].isin(cars_bike_join['walcycdata_id'].unique())]
            # cars_bikes_osm.to_file("shp_files/pr_data.gpkg", layer='cars_bikes_osm_join', driver="GPKG")

        if local_params['bikes and car']:
            print('join  data with bikes and cars count to OSM network')
            osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='car_osm')
            count_osm(osm_data, 3, 'Cyclecount_id')
