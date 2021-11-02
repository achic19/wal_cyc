import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy
import networkx as nx
import osmnx as ox
from geopandas import GeoDataFrame, GeoSeries
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
    cat_file = pd.read_csv('shp_files/cat.csv')
    osm_df = osm_df[osm_df["highway"].isin(cat_file['categories'])]
    cat_file.set_index('categories', inplace=True)
    osm_df["osm_fclass"] = cat_file.loc[osm_df["highway"]]['values'].values
    osm_df["osm_fclass_hir"] = cat_file.loc[osm_df["highway"]]['hierarchical values'].values

    # bool values  -bridge,tunnel
    osm_df = to_bool(osm_df, 'osm_bridge')
    osm_df = to_bool(osm_df, 'osm_tunnel')

    # new fields, walcycdata_id = a unique number with leading 4 for osm
    osm_df.reset_index(inplace=True)
    n = len(osm_df)
    osm_df['walcycdata_id'], osm_df['walcycdata_is_valid'], osm_df['walcycdata_last_modified'] = pd.Series(
        map(lambda x: int('4' + x), np.arange(
            n).astype(str))), 1, date
    # drop fields
    osm_df.drop(columns=['index', 'u', 'v', 'key'], inplace=True)

    # more changes should be aplie
    to_int8 = ['walcycdata_is_valid', 'osm_bridge', 'osm_tunnel', "osm_oneway"]
    osm_df[to_int8] = osm_df[to_int8].astype(np.int8)

    to_int = ['osm_maxspeed', "osm_layer"]
    osm_df[to_int] = osm_df[to_int].apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0).astype(int)

    osm_df.set_index('walcycdata_id', inplace=True)
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
        columns={"OBJECTID": "incident_id", "Land": "incident_land", "Administ_1": "incident_region",
                 "Administra": "incident_district", "Year_of_ac": "incident_year", "Month_of_a": "incident_month",
                 "Day_of_wee": "incident_week_day", "Hour_of_ac": "incident_hour",
                 "Pernosal_i": "incident_personal_injury", "Kinds_of_a": "incident_class",
                 "Type_of_ac": "incident_type", "Light_cond": "incident_light_condition",
                 "Bicycle": "incident_with_bycycle", "Passenger_": "incident_with_passenger_car",
                 "Passenger": "incident_with_passenger", "Motorcycle": "incident_with_motorcyle",
                 "Goods_vehi": "incident_with_goods", "Other": "incident_with_other",
                 "Surface_co": "incident_surface_condition", "LINREFX": "incident_ref_x", "LINREFY": "incident_ref_y",
                 "XGCSWGS84": "incident_gcswgs84_x", "YGCSWGS84": "incident_gcswgs84_y"}, inplace=True)
    incident[
        ['incident_with_bycycle', 'incident_with_passenger_car', 'incident_with_passenger', 'incident_with_motorcyle',
         'incident_with_goods', 'incident_with_other']] = incident[
        ['incident_with_bycycle', 'incident_with_passenger_car', 'incident_with_passenger', 'incident_with_motorcyle',
         'incident_with_goods', 'incident_with_other']].astype(int)
    incident[["incident_ref_x", "incident_ref_y", "incident_gcswgs84_x", "incident_gcswgs84_y"]] = incident[[
        "incident_ref_x", "incident_ref_y", "incident_gcswgs84_x", "incident_gcswgs84_y"]].astype(float)

    # drop fields
    incident.drop(
        columns=['Municipali', 'severity'], inplace=True)

    # new fields, , walcycdata_id = a unique number with leading 1 for incidents
    incident.reset_index(inplace=True)
    n = len(incident)
    incident['walcycdata_id'], incident['walcycdata_is_valid'], incident['walcycdata_last_modified'], incident[
        'osm_walcycdata_id'] = pd.Series(map(lambda x: int('1' + x), np.arange(n).astype(str))), 1, date, -1
    incident = general_tasks(file=incident, is_type=False)
    incident.set_index('walcycdata_id', inplace=True)
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
    count = general_tasks(file=count, is_type=True)
    # change table names
    count.rename(
        columns={"NO": "cyclecount_id", "ZAEHLUN~19": "cyclecount_count"}, inplace=True)

    # drop fields
    count = count[['FROMNODENO', 'TONODENO', 'cyclecount_id', 'cyclecount_count', 'TSYSSET', 'geometry']]

    # new fields, walcycdata_id = a unique number with leading 2 for cycle count
    count.reset_index(inplace=True)
    n = len(count)
    count['walcycdata_id'] = pd.Series(map(lambda x: int('2' + x), np.arange(n).astype(str)))
    count['walcycdata_is_valid'], count['walcycdata_last_modified'], \
    count['osm_walcycdata_id'], count['cyclecount_timestamp_start'], count[
        'cyclecount_timestamp_end'] = 1, date, -1, -1, -1
    count.set_index('walcycdata_id', inplace=True)
    return count.drop('index', axis=1)


def car_count_def(count):
    # project and clip
    count = general_tasks(file=count, is_type=True)
    # change table names
    count.rename(
        columns={"NO": "carcount_id", "ZOhlung_~7": "carcount_count"}, inplace=True)

    # drop fields
    count = count[['FROMNODENO', 'TONODENO', 'carcount_id', 'carcount_count', 'TSYSSET', 'geometry']]

    # new fields, walcycdata_id = a unique number with leading 3 for cycle count
    count.reset_index(inplace=True)
    n = len(count)
    count['walcycdata_id'], count['walcycdata_is_valid'], count['walcycdata_last_modified'], \
    count['osm_walcycdata_id'], count['carcount_timestamp_start'], count[
        'carcount_timestamp_end'] = pd.Series(
        map(lambda x: int('3' + x), np.arange(n).astype(str))), 1, date, -1, -1, -1
    count.set_index('walcycdata_id', inplace=True)
    return count.drop('index', axis=1)


def general_tasks(file: GeoDataFrame, is_type: bool) -> GeoDataFrame:
    """
    :param file: delete nan geometry and than reprojected, clip and do it unidirectional
    :param is_type: control whether to remove certain types
    :return: gdf
    """
    print("gis_oper_for_all")
    file = file[~file.is_empty]
    clipped = gpd.clip(file.to_crs(crs), clip_file)
    clipped = clipped[~clipped.is_empty]
    if is_type:
        clipped['is_type'] = clipped['TSYSSET'].apply(remove_type)
        return clipped[clipped['is_type']]
    else:
        return clipped


def remove_type(value):
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
        if val not in type_to_remove:
            return True
    return False


def overlay_count_osm(osm_gdf_0: GeoDataFrame, local_network: GeoDataFrame, osm_columns: list,
                      local_columns: list, dissolve_by: str) -> dict:
    """
    This function calculate between osm entities and local network entities
    :param osm_gdf_0: The OSM network to work on
    :param local_network: count data on local network
    :param local_columns: fields of local network to save while the matching process
    :param osm_columns: fields of osm network to save while the matching process
    :return: list of all the gis files been created during implementation of this stage
    :param dissolve_by: before buffering, the code dissolve osm entities by that field
    """

    # make buffer of 10 meters around each polyline in the both dataframes
    # and calculate the overlay intersection between the two.
    print('osm_dissolve')
    osm_gdf = osm_gdf_0.dissolve(by=dissolve_by).reset_index()

    print('azimuth')
    # osm_gdf['azimuth'] = osm_gdf['geometry'].apply(calculate_azimuth)
    # local_network['azimuth'] = local_network['geometry'].apply(calculate_azimuth)

    print('osm_buffer')
    osm_buffer = GeoDataFrame(osm_gdf[osm_columns],
                              geometry=osm_gdf.geometry.buffer(15, cap_style=2), crs=osm_gdf.crs)

    print('count_buffer')
    count_buffer = gpd.GeoDataFrame(local_network[local_columns], crs=local_network.crs,
                                    geometry=local_network.geometry.buffer(15, cap_style=2))

    print('overlay')
    overlay = count_buffer.overlay(osm_buffer, how='intersection')
    # Calculate the percentage field
    print('percentage')
    overlay['areas'] = overlay.area
    count_buffer['areas'] = count_buffer.area
    # The index column in the overlay layer contains the id of the count entity,
    # therefore in order to calculate the rational area between the overlay polygon and the corresponding polygon,
    # the index field becomes the index. In order to save the percentage results,
    # the table is sorted by index and area and then the index is reset.
    overlay.set_index('index', inplace=True)
    temp = (overlay['areas'] * 100 / count_buffer[count_buffer['index'].isin(overlay.index)].set_index('index')[
        'areas']).reset_index()
    temp = temp.sort_values(['index', 'areas'])['areas']
    overlay = overlay.reset_index().sort_values(['index', 'areas'])
    overlay['percentage'] = temp.values

    return {'overlay': overlay, 'osm_buffer': osm_buffer,
            'count_buffer': count_buffer, 'osm_gdf': osm_gdf}


def calculate_percentage(row, count_buffer):
    return row['area'] / count_buffer[count_buffer['index'] == row['index']]['area']


def calculate_azimuth(row):
    geo = row.coords
    return math.degrees(math.atan2(geo[-1][0] - geo[0][0], geo[-1][1] - geo[0][1])) % 360


def map_matching(overlay: GeoDataFrame, file_count_to_update: GeoDataFrame, groupby_field: str) -> GeoDataFrame:
    """
    map between the local network to the osm network based on the largest overlay between two entities
    :param overlay: polygons of overlay
    :param file_count_to_update:
    :param groupby_field:
    :return:
    """
    print('start map matching')
    # for each count object return the osm id with the larger overlay with him
    matching_info = overlay.groupby(groupby_field).apply(find_optimal_applicant)
    matching_info = matching_info[matching_info.notna()]
    # update cycle count data with the corresponding osm id
    matching_info.sort_index(inplace=True)

    file_count_to_update.set_index(groupby_field, drop=False, inplace=True)
    file_count_to_update.sort_index(inplace=True)
    file_count_to_update['osm_walcycdata_id'] = -1
    file_count_to_update['osm_walcycdata_id'][
        file_count_to_update[groupby_field].isin(matching_info.index)] = matching_info['osm_id']
    file_count_to_update.drop('index', axis=1, inplace=True)
    print('finish map matching')
    return file_count_to_update


def find_optimal_applicant(group):
    """
    By analyzing several conditions, this function finds the best candidates out of all the options:
    1. The highest in the hierarchy should be the first ti check
    2. Then these with the largest overlay should be chosen
    3. Select the applicant if he meets the parallel condition and the minimum overlay
    :param group: group of applicants (osm entities)
    :return:
    """
    if len(group) > 1:
        group.sort_values(['osm_fclass_hir', 'areas'], ascending=False, inplace=True)
        for row in group.iterrows():
            if is_parallel_more_than_min(row[1]):
                return row[1]

    else:
        row = group.iloc[0]
        if is_parallel_more_than_min(row):
            return row


def is_parallel_more_than_min(row) -> bool:
    """
   Using this function you can check if the lines have more or less of the same direction and
   if the overlap between their polygons is more than 20%
    :param row:
    :return:
    """
    # if row['parallel'] < 30 and row['percentage'] > 20:

    if row['percentage'] > 20:
        return True
    else:
        return False


if __name__ == '__main__':
    # general code
    clip_file = gpd.read_file('shp_files/munich_3857.shp')
    crs = "EPSG:3857"
    date = pd.to_datetime("today")
    type_to_remove = ['C', 'D', 'NT', 'S', 'T', 'U']
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')

    params = {'osm': [False, {'prepare_osm_data': False, 'osm_file': True, 'car_bike_osm': False}],
              'count': [False,
                        {'cycle_count': False, 'car_count': False, 'merge_files': True, 'delete_null_zero': False}],
              'incident': [False, {'prepare_incident': True, 'join_to_bike_network': False}],
              'count_osm': [True, {'prepare_overlay': True, 'matching': False}],
              'data_to_server': [False, {'osm': False, 'bikes': False, 'cars': True, 'incidents': False}]}

    if params['osm'][0]:
        local_params = params['osm'][1]
        if local_params['prepare_osm_data']:
            prepare_osm_data()
        if local_params['osm_file']:
            print('create osm table')
            my_osm_file = osm_file_def()
            my_osm_file.to_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network', driver="GPKG")

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

        if local_params['car_count']:
            print('create car_count table')
            car_count = gpd.read_file('shp_files/cars_count_data.shp')
            my_car_count = car_count_def(car_count)
            my_car_count.to_file("shp_files/pr_data.gpkg", layer='car_count', driver="GPKG")

        if local_params['merge_files']:
            print('merge file')
            cycle_count = gpd.read_file("shp_files/pr_data.gpkg", layer='cycle_count')[
                ['walcycdata_id', 'cyclecount_id', 'cyclecount_count', 'geometry']]
            car_count = gpd.read_file("shp_files/pr_data.gpkg", layer='car_count')[
                ['walcycdata_id', 'carcount_id', 'carcount_count', 'geometry']]

            cycle_count.drop_duplicates(subset=['cyclecount_id'], inplace=True)
            car_count.drop_duplicates(subset=['carcount_id'], inplace=True)

            all_count = cycle_count.merge(right=car_count, left_on='cyclecount_id',
                                          right_on='carcount_id', how='outer', suffixes=('_cycle', '_car'))
            # 'geometry_cycle' will be the geometry for all the the entities in the new file
            all_count.loc[all_count['geometry_cycle'].isnull(), 'geometry_cycle'] = \
                all_count[all_count['geometry_cycle'].isnull()]['geometry_car']
            all_count = GeoDataFrame(all_count, geometry=all_count['geometry_cycle'], crs=crs)

            all_count.drop(['geometry_car', 'geometry_cycle'], axis=1, inplace=True)
            all_count.reset_index().to_file("shp_files/pr_data.gpkg", layer='all_count', driver="GPKG")
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

    if params['count_osm'][0]:
        local_params = params['count_osm'][1]
        count_data = gpd.read_file("shp_files/pr_data.gpkg", layer='all_count')
        if local_params['prepare_overlay']:
            print(' start overlay')
            osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network')
            osm_data_columns = ['osm_id', 'highway', 'osm_fclass_hir']
            count_columns = ['index', 'walcycdata_id_cycle', 'walcycdata_id_car', 'cyclecount_count', 'carcount_count']
            results = overlay_count_osm(osm_gdf_0=osm_data, local_network=count_data, osm_columns=osm_data_columns,
                                        local_columns=count_columns, dissolve_by='osm_id')
            print('finish overlay')
            [item[1].to_file("shp_files/matching_files.gpkg", layer=item[0]) for item in results.items()]
        if local_params['matching']:
            osm_munich_overlay = gpd.read_file('shp_files/matching_files.gpkg', layer='overlay')
            map_matching(overlay=osm_munich_overlay, file_count_to_update=count_data, groupby_field='index').to_file(
                "shp_files/matching_files.gpkg", layer='count_osm_matching')

    if params['data_to_server'][0]:
        local_params = params['data_to_server'][1]
        if local_params['osm']:
            pass
        if local_params['bikes']:
            print("upload cycle_count_data")
            my_cycle_count = gpd.read_file("shp_files/pr_data.gpkg", layer='cycle_count')
            my_cycle_count.to_postgis(name="cycle_count_data", con=engine, schema='production',
                                      if_exists='replace',
                                      dtype={'walcycdata_last_modified': sqlalchemy.types.DateTime})
        if local_params['cars']:
            print("upload car_count_data")
            my_car_count = gpd.read_file("shp_files/pr_data.gpkg", layer='car_count', driver="GPKG")
            my_car_count.to_postgis(name="car_count_data", con=engine, schema='production',
                                    if_exists='replace',
                                    dtype={'walcycdata_last_modified': sqlalchemy.types.DateTime})
        if local_params['incidents']:
            pass
