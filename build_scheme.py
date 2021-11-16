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


def cycle_count_def(count, counts_nodes: GeoDataFrame):
    """
    This method calcultes
    :param count:
    :param counts_nodes: for Azimuth calculation
    :return:
    """
    # project and clip
    count = general_tasks(file=count, is_type=True, counts_nodes=counts_nodes)
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
        'cyclecount_timestamp_end'] = 1, date, -1, -1, -1
    count.set_index('walcycdata_id', inplace=True)
    return count.drop('index', axis=1)


def car_count_def(count, counts_nodes: GeoDataFrame):
    # project and clip
    count = general_tasks(file=count, is_type=True, counts_nodes=counts_nodes)
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
        map(lambda x: int('3' + x), np.arange(n).astype(str))), 1, date, -1, -1, -1
    count.set_index('walcycdata_id', inplace=True)
    return count.drop('index', axis=1)


def general_tasks(file: GeoDataFrame, is_type: bool, counts_nodes: GeoDataFrame) -> GeoDataFrame:
    """
    :param file: delete nan geometry and than reprojected, clip and do it unidirectional
    :param is_type: control whether to remove certain types
    :return: gdf
    :param counts_nodes: for Azimuth calculation
    """
    print("_gis_oper_for_all")
    print('__clip')
    file = file[~file.is_empty]
    # "Sjoin" is used to remove the entire segment that intersects the clip.
    clipped = file.to_crs(crs).sjoin(clip_file, how="inner", predicate='within')
    clipped = clipped[~clipped.is_empty][file.columns]
    # calculate azimuth
    print("__calculate azimuth")
    counts_nodes.set_index('NO', inplace=True)
    clipped['azimuth'] = clipped.apply(
        lambda row: calculate_azimuth(counts_nodes.loc[row['FROMNODENO']], counts_nodes.loc[row['TONODENO']]), axis=1)
    if is_type:
        print("__calculate is_type")
        clipped['is_type'] = clipped['TSYSSET'].apply(remove_type)
        return clipped[clipped['is_type']]
    else:
        return clipped


def calculate_azimuth(p1, p2):
    try:
        return math.degrees(math.atan2(p2['XCOORD'] - p1['XCOORD'], p2['YCOORD'] - p1['YCOORD'])) % 360
    except:
        print('can not calculate azimuth')
        return -1


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
    osm_gdf['azimuth'] = osm_gdf[dissolve_by].apply(azimuth_osm)

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
    print('calculate angles between elements')
    overlay['parallel'] = overlay.apply(lambda x: angle_between(x, local_network, osm_gdf), axis=1)
    return {'overlay': overlay, 'osm_buffer': osm_buffer,
            'count_buffer': count_buffer, 'osm_gdf': osm_gdf}


def angle_between(row, local_network, osm_gdf):
    """
    This function add the angle between osm element and local element (the names of these elements are stored in row) and
    azimuth in local_network and osm_gdf
    :param osm_gdf:
    :param local_network:
    :param row:
    :return:
    """
    local_az = local_network[local_network['index'] == row['index']]['azimuth'].values[0]
    osm_az = osm_gdf[osm_gdf['osm_id'] == row['osm_id']]['azimuth'].values[0]
    if local_az == -1 or osm_az == -1:
        return -1
    else:
        return abs(local_az - osm_az) % 180 % 150


def azimuth_osm(osm_id):
    pnt_0 = nodes[ways[osm_id][0]]
    pnt_1 = nodes[ways[osm_id][-1]]
    if pnt_0 == pnt_1:
        return -1
    else:
        return math.degrees(math.atan2(pnt_1[0] - pnt_0[0], pnt_1[1] - pnt_0[1])) % 360


def list_str(as_str):
    print('list_str')
    # pd_1 = pd.DataFrame({'index': [680, 122956, 122957], 'areas': [20, 80, 60]})
    as_str = list(map(lambda x: x + ',', as_str))
    as_str = ''.join(as_str)[:-1]
    return as_str


def osm_id_req(id_osm_local):
    print('osm_id_req')
    import overpy

    api = overpy.Overpass()
    overpass_query = """[out:json];area[name="München"];way(id:""" + id_osm_local + """); out geom;"""
    result = api.query(overpass_query)
    return list(map(lambda x: calc_azi(x.attributes['geometry']), result.ways))


def calc_azi(geom):
    print('calc_azi')
    import math
    return math.degrees(math.atan2(geom[-1]['lon'] - geom[0]['lon'], geom[-1]['lat'] - geom[0]['lat'])) % 360


def calculate_percentage(row, count_buffer):
    return row['area'] / count_buffer[count_buffer['index'] == row['index']]['area']


def map_matching(overlay: GeoDataFrame, file_count_to_update: GeoDataFrame, groupby_field: str) -> GeoDataFrame:
    """
    map between the local network to the osm network based on the largest overlay between two entities
    :param overlay: polygons of overlay
    :param file_count_to_update:
    :param groupby_field:
    :return:
    """
    print('start map matching')
    print(' find the best matching')
    # for each count object return the osm id with the larger overlay with him
    matching_info = overlay.groupby(groupby_field).apply(find_optimal_applicant)
    print(' finish the best matching')
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
    4. in case of more no one is match the
    :param group: group of applicants (osm entities)
    :return:
    """
    if len(group) > 1:
        group.sort_values(['osm_fclass_hir', 'areas'], ascending=False, inplace=True)
        for row_temp in group.iterrows():
            row = row_temp[1]
            if is_parallel_more_than_min(row):
                row['valid'] = True
                return row
        row = group.iloc[0]
        row['valid'] = False
        return row
    else:
        row = group.iloc[0]
        if is_parallel_more_than_min(row):
            row['valid'] = True
            return row
        else:
            row['valid'] = False
            return row


def is_parallel_more_than_min(row) -> bool:
    """
   Using this function you can check if the lines have more or less of the same direction and
   if the overlap between their polygons is more than 20%
    :param row:
    :return:
    """

    if row['parallel'] < 30 and row['percentage'] > 20:
        return True
    else:
        return False


def find_the_azimuth_between_two_options(az_1, az_2):
    """
    The method returns an azimuth that is not zero. The two values should be the same if they are not zero.
    :param az_1
    :param az_2
    :return:
    """

    if az_1 != 0:
        if az_2 != 0:
            if round(az_1) == round(az_2):
                return az_1
            else:
                return -2
        else:
            return az_1
    else:
        if az_2 != 0:
            return az_2
        else:
            return -1


if __name__ == '__main__':

    # Get the the original OSM network without any editing
    import pickle

    with open('response_jsons.pkl', 'rb') as response_jsons_file:
        response_jsons = pickle.load(response_jsons_file)
    ways = {way['id']: way['nodes'] for way in response_jsons[0]['elements'] if way['type'] == 'way'}
    nodes = {node['id']: (node['lon'], node['lat']) for node in response_jsons[0]['elements'] if
             node['type'] == 'node'}

    # global variables
    clip_file = gpd.read_file('shp_files/munich_3857.shp')
    crs = "EPSG:3857"
    date = pd.to_datetime("today")
    type_to_remove = ['C', 'D', 'NT', 'S', 'T', 'U']
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')
    # dictionary to control which function to run
    params = {'osm': [False, {'prepare_osm_data': True, 'osm_file': True, 'car_bike_osm': False}],
              'count': [True,
                        {'cycle_count': False, 'car_count': False, 'merge_files': True}],
              'incident': [False, {'prepare_incident': True, 'join_to_bike_network': False}],
              'count_osm': [False, {'prepare_overlay': True, 'matching': True}],
              'analysis': False,
              'data_to_server': [False, {'osm': False, 'bikes': False, 'cars': True, 'incidents': False}]}

    if params['osm'][0]:
        # Prepare OSM information
        local_params = params['osm'][1]
        if local_params['prepare_osm_data']:
            prepare_osm_data()
        if local_params['osm_file']:
            print('create osm table')
            my_osm_file = osm_file_def()
            my_osm_file.to_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network', driver="GPKG")

    if params['count'][0]:
        local_params = params['count'][1]

        if local_params['cycle_count']:
            print('create cycle_count table')
            # upload files
            cycle_count = gpd.read_file('shp_files/cycle_count_data.shp')
            cycle_count_nodes = gpd.read_file('shp_files/cycle_count_data_nodes.shp')
            my_cycle_count = cycle_count_def(count=cycle_count, counts_nodes=cycle_count_nodes)
            print('write to disk')
            my_cycle_count.to_file("shp_files/pr_data.gpkg", layer='cycle_count', driver="GPKG")

        if local_params['car_count']:
            print('create car_count table')
            car_count = gpd.read_file('shp_files/cars_count_data.shp')
            car_count_nodes = gpd.read_file('shp_files/car_count_data_nodes.shp')
            my_car_count = car_count_def(car_count, car_count_nodes)
            my_car_count.to_file("shp_files/pr_data.gpkg", layer='car_count', driver="GPKG")

        if local_params['merge_files']:
            print('merge file')
            cycle_count = gpd.read_file("shp_files/pr_data.gpkg", layer='cycle_count')[
                ['walcycdata_id', 'cyclecount_id', 'cyclecount_count', 'azimuth', 'FROMNODENO', 'TONODENO', 'geometry']]
            car_count = gpd.read_file("shp_files/pr_data.gpkg", layer='car_count')[
                ['walcycdata_id', 'carcount_id', 'carcount_count', 'azimuth', 'FROMNODENO', 'TONODENO', 'geometry']]

            cycle_count.drop_duplicates(subset=['cyclecount_id'], inplace=True)
            car_count.drop_duplicates(subset=['carcount_id'], inplace=True)

            all_count = cycle_count.merge(right=car_count, left_on=['FROMNODENO', 'TONODENO'],
                                          right_on=['FROMNODENO', 'TONODENO'], how='outer', suffixes=('_cycle', '_car'))
            # 'geometry_cycle' will be the geometry for all the the entities in the new file
            all_count.loc[all_count['geometry_cycle'].isnull(), 'geometry_cycle'] = \
                all_count[all_count['geometry_cycle'].isnull()]['geometry_car']
            all_count = GeoDataFrame(all_count, geometry=all_count['geometry_cycle'], crs=crs)

            all_count.drop(['geometry_car', 'geometry_cycle'], axis=1, inplace=True)
            # Combine two azimuth columns into one
            all_count[['azimuth_car', 'azimuth_cycle']] = all_count[
                ['azimuth_car', 'azimuth_cycle']].fillna(0)
            print('_find_the_azimuth_between_two_options')
            all_count['azimuth'] = all_count.apply(
                lambda row: find_the_azimuth_between_two_options(row['azimuth_cycle'], row['azimuth_car']), axis=1)
            print('write to disk')
            all_count.reset_index().to_file("shp_files/pr_data.gpkg", layer='all_count', driver="GPKG")

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
            print('start overlay')
            osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network')
            osm_data_columns = ['osm_id', 'highway', 'osm_fclass_hir', 'azimuth']
            count_columns = ['index', 'walcycdata_id_cycle', 'walcycdata_id_car', 'cyclecount_count', 'carcount_count',
                             'azimuth']
            results = overlay_count_osm(osm_gdf_0=osm_data, local_network=count_data, osm_columns=osm_data_columns,
                                        local_columns=count_columns, dissolve_by='osm_id')
            print('finish overlay')
            [item[1].to_file("shp_files/matching_files.gpkg", layer=item[0]) for item in results.items()]
        if local_params['matching']:
            osm_munich_overlay = gpd.read_file('shp_files/matching_files.gpkg', layer='overlay')
            map_matching(overlay=osm_munich_overlay, file_count_to_update=count_data, groupby_field='index').to_file(
                "shp_files/matching_files.gpkg", layer='count_osm_matching')
    if params['analysis']:
        print('analysis')
        from classes.Analysis import Analysis

        cat_file = pd.read_csv('shp_files/cat.csv')
        osm_data = gpd.read_file("shp_files/matching_files.gpkg", layer='osm_gdf')
        matching_data = gpd.read_file("shp_files/matching_files.gpkg", layer='count_osm_matching')[
            'osm_walcycdata_id'].unique()
        results = Analysis.osm_network_local_network(osm_network=osm_data, osm_id_list=matching_data, cat_file=cat_file)
        results.to_file("shp_files/matching_files.gpkg", layer='osm_linked_to_counting', driver="GPKG")

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
