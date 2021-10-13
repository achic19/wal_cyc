import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import networkx as nx
import osmnx as ox
from geopandas import GeoDataFrame


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
    area = gpd.read_file(r'shp_files\munich_4326.shp')['geometry'][0]
    graph = ox.graph_from_polygon(area, network_type='all')
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
    n = len(incident)
    incident['walcycdata_id'], incident['walcycdata_is_valid'], incident['walcycdata_last_modified'], incident[
        'osm_walcycdata_id'] = pd.Series(map(lambda x: int('1' + x), np.arange(n).astype(str))), True, date, ''
    incident = gis_opers_for_all(file=incident, unidirectional=False)
    return incident


def join_to_bike_network(incidentgdb: GeoDataFrame, network: GeoDataFrame) -> GeoDataFrame:
    """
    Find for each incident the closet street segment (with bike 'NO')
    :param incidentgdb:
    :param network:
    :return:
    """

    # remove from the network rows with Null values in 'Cyclecount_id'
    network = GeoDataFrame(network[network['Cyclecount_id'].notna()]['Cyclecount_id'], geometry=network.geometry,
                           crs=network.crs)
    return gpd.sjoin_nearest(incidentgdb, network, how='left', distance_col='dist')


def cycle_count_def(count):
    # project and clip
    count = gis_opers_for_all(file=count, count_column="ZAEHLUN~19", unidirectional=True)
    # change table names
    count.rename(
        columns={"NO": "Cyclecount_id", "count": "Cyclecount_count"}, inplace=True)

    # drop fields
    count = count[['Cyclecount_id', 'Cyclecount_count', 'geometry']]

    # new fields, walcycdata_id = a unique number with leading 2 for cycle count
    n = len(count)
    count['walcycdata_id'] = pd.Series(map(lambda x: int('2' + x), np.arange(n).astype(str)))
    count['walcycdata_is_valid'], count['walcycdata_last_modified'], \
    count['osm_walcycdata_id'], count['Cyclecount_timestamp_start'], count[
        'Cyclecount_timestamp_end'] = True, date, '', '', ''
    return count


def car_count_def(count):
    # project and clip
    count = gis_opers_for_all(file=count, count_column="ZOhlung_~7", unidirectional=True)
    # change table names
    count.rename(
        columns={"NO": "Carcount_id", "count": "Carcount_count"}, inplace=True)

    # drop fields
    count = count[['Carcount_id', 'Carcount_count', 'geometry']]

    # new fields, walcycdata_id = a unique number with leading 3 for cycle count
    n = len(count)
    count['walcycdata_id'], count['walcycdata_is_valid'], count['walcycdata_last_modified'], \
    count['osm_walcycdata_id'], count['Carcount_timestamp_start'], count[
        'Carcount_timestamp_end'] = pd.Series(
        map(lambda x: int('3' + x), np.arange(n).astype(str))), True, date, '', '', ''
    return count


def gis_opers_for_all(file: GeoDataFrame, unidirectional: bool, count_column: str = '') -> GeoDataFrame:
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
        # make it unidirectional grouping by "NO'(sum) and than drop_duplicates 'ON'
        clipped.fillna(0, inplace=True)
        group_by = clipped.groupby(['NO']).sum()
        clipped.drop_duplicates(subset=['NO'], inplace=True)
        clipped.set_index('NO', inplace=True)
        clipped['count'] = group_by[count_column]
        return clipped.reset_index()
    else:
        return clipped


def osm_file_def():
    osm_file = gpd.read_file('shp_files/osm_road_network.shp')

    # drop duplicate
    osm_file.drop_duplicates(subset=['geometry'], inplace=True)
    # change table names
    osm_file.rename(
        columns={"osmid": "osm_id", "name": "osm_name", "oneway": "osm_oneway",
                 "maxspeed": "osm_maxspeed",
                 "layer": "osm_layer", "bridge": "osm_bridge", "tunnel": "osm_tunnel"}, inplace=True)
    # categorical values  - f_class
    cat_file = pd.read_csv('shp_files/cat.csv', header=None, names=['values', 'categories'])
    osm_file = osm_file[osm_file["highway"].isin(cat_file['categories'])]
    cat_file.set_index('categories', inplace=True)
    osm_file["osm_fclass"] = cat_file.loc[osm_file["highway"]]['values'].values

    # null in layer = ground level = 0

    osm_file["osm_layer"].fillna(0, inplace=True)
    # bool values  -bridge,tunnel
    osm_file = to_bool(osm_file, 'osm_bridge')
    osm_file = to_bool(osm_file, 'osm_tunnel')
    # new fields, walcycdata_id = a unique number with leading 4 for osm
    n = len(osm_file)
    osm_file['walcycdata_id'], osm_file['walcycdata_is_valid'], osm_file['walcycdata_last_modified'] = pd.Series(
        map(lambda x: int('4' + x), np.arange(
            n).astype(str))), True, date

    return osm_file


def to_bool(osm_file, field):
    osm_file[field].loc[~osm_file[field].isnull()] = 1  # not nan
    osm_file[field].loc[(osm_file[field].isnull()) | (osm_file[field] == 'no')] = 0  # not nan
    return osm_file


if __name__ == '__main__':
    # general code
    clip_file = gpd.read_file('shp_files/munich_3857.shp')
    crs = "EPSG:3857"
    date = pd.to_datetime("today")
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')

    params = {'osm': [True, {'prepare_osm_data': False, 'osm_file': True, 'car_bike_osm': True}],
              'count': [False,
                        {'cycle_count': False, 'car_count': False, 'merge_files': False, 'delete_null_zero': True}],
              'incident': [False, {'prepare_incident': False, 'join_to_bike_network': False}],
              'count_osm': [True, {'bikes': False, 'cars': True}],
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
            osm_file = gpd.read_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network')
            bike_osm = osm_file.set_index('highway').drop(labels=['footway', 'steps'], axis=0)
            car_osm = bike_osm.drop(labels=['cycleway', 'bridleway'], axis=0)
            bike_osm.reset_index().to_file("shp_files/pr_data.gpkg", layer='cycle_osm', driver="GPKG")
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
        if local_params['cars']:
            osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='car_osm')
            count_data_cars = count_data[count_data['Carcount_count']>0]

