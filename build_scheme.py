import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, MetaData
import sqlalchemy
from migrate.changeset.constraint import PrimaryKeyConstraint

import networkx as nx
import osmnx as ox
from osmnx import downloader
from geopandas import GeoDataFrame, GeoSeries
from shapely.coords import CoordinateSequence
import math
from classes.osm import *
from classes.count_osm_matching import *
from classes.counting import *
from classes.incidents import *


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
    # ToDo should be update
    # incident['walcycdata_id'], incident['walcycdata_is_valid'], incident['walcycdata_last_modified'], incident[
    #     'osm_walcycdata_id'] = pd.Series(map(lambda x: int('1' + x), np.arange(n).astype(str))), 1, date, -1
    # incident = general_tasks(file=incident, is_type=False)
    incident.set_index('walcycdata_id', inplace=True)
    return incident


def merge_users_network(cycle_merge, car_merge) -> GeoDataFrame:
    """
    merge between two files based on the fields in fields_merge
    :param cycle_merge:
    :param car_merge:
    :return:
    """

    all_merge = cycle_merge.merge(right=car_merge, left_on=['FROMNODENO', 'TONODENO'],
                                  right_on=['FROMNODENO', 'TONODENO'], how='outer', suffixes=('_cycle', '_car'))
    # 'geometry_cycle' will be the geometry for all the the entities in the new file
    all_merge.loc[all_merge['geometry_cycle'].isnull(), 'geometry_cycle'] = \
        all_merge[all_merge['geometry_cycle'].isnull()]['geometry_car']
    all_merge = GeoDataFrame(all_merge, geometry=all_merge['geometry_cycle'], crs=MunichData.crs)

    all_merge.drop(['geometry_car', 'geometry_cycle'], axis=1, inplace=True)
    # Combine two azimuth columns into one
    all_merge[['azimuth_car', 'azimuth_cycle']] = all_merge[
        ['azimuth_car', 'azimuth_cycle']].fillna(0)
    print('_find_the_azimuth_between_two_options')
    all_merge['azimuth'] = all_merge.apply(
        lambda row: find_the_azimuth_between_two_options(row['azimuth_cycle'], row['azimuth_car']), axis=1)
    return all_merge[all_merge['azimuth'] > -1].drop(columns=['azimuth_car', 'azimuth_cycle'])


def overlay_count_osm(osm_gdf: GeoDataFrame, local_network: GeoDataFrame, osm_columns: list,
                      local_columns: list) -> dict:
    """
    This function calculate between osm entities and local network entities
    :param osm_gdf: The OSM network to work on
    :param local_network: count data on local network
    :param local_columns: fields of local network to save while the matching process
    :param osm_columns: fields of osm network to save while the matching process
    :return: list of all the gis files been created during implementation of this stage
    """

    # and calculate the overlay intersection between the two.

    print('__osm_buffer')
    osm_buffer = GeoDataFrame(osm_gdf[osm_columns],
                              geometry=osm_gdf.geometry.buffer(15, cap_style=2), crs=osm_gdf.crs)

    print('__count_buffer')
    count_buffer = GeoDataFrame(local_network[local_columns], crs=local_network.crs,
                                geometry=local_network.geometry.buffer(15, cap_style=2))

    print('__overlay')
    overlay = count_buffer.overlay(osm_buffer, how='intersection')
    # Calculate the percentage field
    print('__percentage')
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
    print('__calculate angles between elements')
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
    print('_start map matching')
    print('__find the best matching')
    # for each count object return the osm id with the larger overlay with him
    matching_info = overlay.groupby(groupby_field).apply(find_optimal_applicant)
    print('__finish the best matching')
    matching_info = matching_info[matching_info.notna()]
    # update cycle count data with the corresponding osm id
    matching_info.sort_index(inplace=True)

    file_count_to_update.set_index(groupby_field, drop=False, inplace=True)
    file_count_to_update.sort_index(inplace=True)
    file_count_to_update['osm_walcycdata_id'] = -1
    file_count_to_update['valid'] = -1
    file_count_to_update['osm_walcycdata_id'][
        file_count_to_update[groupby_field].isin(matching_info.index)] = matching_info['osm_id']
    file_count_to_update['valid'][
        file_count_to_update[groupby_field].isin(matching_info.index)] = matching_info['valid']
    file_count_to_update.drop('index', axis=1, inplace=True)
    print('_finish map matching')
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
    The method returns an azimuth that is not zero. az_1 is prefared over az_2 as it based on newer network
    :param az_1
    :param az_2
    :return:
    """
    if az_1 != 0:
        return az_1
    elif az_2 != 0:
        return az_2
    else:
        return -1


if __name__ == '__main__':
    # general variables
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')
    clip_file = gpd.read_file('shp_files/munich_3857.shp')
    # dictionary to control which function to run
    params = {'osm': [False, {'prepare_osm_data': False, 'osm_file': False, 'data_to_server': True,
                              'find_the_opposite_roads': False, 'stat_one_way': False}],
              'count': [False,
                        {'cycle_count': False, 'car_count': False, 'merge_files': False, 'data_to_server_car': False,
                         'data_to_server_cycle': True}],
              'incident': [False, {'prepare_incident': False, 'data_to_server': True}],
              'count_osm': [False,
                            {'prepare_overlay': False, 'matching': False, 'two_ways_matching': False, 'refine_matching':
                                False, 'matching_incidents': False, 'matching_to_osm_counting': True,
                             'matching_to_osm_incidents': True
                             }],
              'munich_data': [True, {'prepare_relations': False, 'add_projection_points_to_server': False,
                                     'relations_for_server': True}],
              'analysis': [False, {'osm_network_local_network': False, 'analysis_relations': True}],
              'data_to_server': [False, {'osm': True, 'bikes': False, 'cars': False, 'incidents': False,
                                         'combined_network': False}]}

    if params['osm'][0]:
        print('osm')
        # Prepare OSM information
        local_params = params['osm'][1]
        if local_params['prepare_osm_data']:
            # Here the data is downloaded from OSM server to the local machine
            polygon = gpd.read_file(r'shp_files\munich_4326_large.shp')['geometry'][0]
            lines, pnts = OSM.prepare_osm_data(polygon=polygon)
            lines.to_file("shp_files/osm/osm.gpkg", layer='openstreetmap_data',
                          driver="GPKG")
            pnts.to_file("shp_files/osm/osm.gpkg", layer='openstreetmap_data_nodes',
                         driver="GPKG")
        if local_params['osm_file']:
            # The data is prepared for production
            print('create osm table')
            osm_df = gpd.read_file("shp_files/osm/osm.gpkg", layer='openstreetmap_data')
            cat_file = pd.read_csv('shp_files/osm/cat.csv')
            OSM.osm_file_def(osm_df, cat_file).to_file("shp_files/osm/osm.gpkg", layer='openstreetmap_road_network',
                                                       driver="GPKG")

        osm_data = gpd.read_file("shp_files/osm/osm.gpkg", layer='openstreetmap_road_network')
        if local_params['data_to_server']:
            osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network_final')
            print('_data_to_server')
            OSM.data_to_server(data_to_upload=osm_data, columns_to_upload=OSM.osm_column_names,
                               table_name='openstreetmap_road_network', engine=engine)
        if local_params['find_the_opposite_roads']:
            # The algorithm links two-way roads
            OSM.find_the_opposite_roads(osm_gdf=osm_data).to_file("shp_files/osm/osm.gpkg",
                                                                  layer='openstreetmap_road_network', driver="GPKG")
        if local_params['stat_one_way']:
            osm_data = gpd.read_file("shp_files/osm/osm.gpkg", layer='openstreetmap_road_network')
            print('stat for one way')
            res = OSM.statistic(osm_data)
            file_name = '"shp_files/osm/stat_one_way.csv'
            res.to_csv(file_name)

    if params['count'][0]:
        local_params = params['count'][1]

        if local_params['cycle_count']:
            print('create cycle_count table')
            # upload files
            cycle_count = gpd.read_file('shp_files/cycle_counting.gpkg', layer='Radnetz_VVD-M_Analyse_2019_link',
                                        driver="GPKG")
            cycle_count_nodes = gpd.read_file('shp_files/cycle_count', layer='Radnetz_VVD-M_Analyse_2019_node',
                                              driver="GPKG")

            my_cycle_count = CycleCount.cycle_count_def(count=cycle_count, counts_nodes=cycle_count_nodes,
                                                        clip_file=clip_file)

            print('write to disk')
            my_cycle_count.to_file("shp_files/pr_data.gpkg", layer='cycle_count', driver="GPKG")

        if local_params['car_count']:
            print('create car_count table')
            car_count = gpd.read_file('shp_files/cars_count_data.shp')
            car_count_nodes = gpd.read_file('shp_files/car_count_data_nodes.shp')
            my_car_count = CarCount.car_count_def(count=car_count, counts_nodes=car_count_nodes,
                                                  clip_file=clip_file)
            my_car_count.to_file("shp_files/pr_data.gpkg", layer='car_count', driver="GPKG")

        if local_params['merge_files']:
            print('merge file')
            cycle = gpd.read_file("shp_files/pr_data.gpkg", layer='cycle_count')[
                ['walcycdata_id', 'cyclecount_id', 'cyclecount_count', 'azimuth', 'FROMNODENO', 'TONODENO', 'geometry']]
            car = gpd.read_file("shp_files/pr_data.gpkg", layer='car_count')[
                ['walcycdata_id', 'carcount_id', 'carcount_count', 'azimuth', 'FROMNODENO', 'TONODENO', 'geometry']]
            all_count = merge_users_network(cycle_merge=cycle, car_merge=car)
            print('_write to disk')
            all_count.reset_index().to_file("shp_files/pr_data.gpkg", layer='all_count', driver="GPKG")
        if local_params['data_to_server_car']:
            print("upload car data")
            data = gpd.read_file("shp_files/pr_data.gpkg", layer='car_count2', driver="GPKG")
            # ToDo should be deleted
            data.rename(columns={'osm_walcycdata_id': 'osm_id'}, inplace=True)
            data['carcount_count'] = data['carcount_count'].fillna(0)
            data['osm_id'] = data['osm_id'].fillna(0)
            Counting.data_to_server(data_to_upload=data, columns_to_upload=CarCount.column_names(),
                                    table_name='car_count_data')
        if local_params['data_to_server_cycle']:
            print("upload cycle data")
            data = gpd.read_file("shp_files/pr_data.gpkg", layer='cycle_count2', driver="GPKG")
            # ToDo should be deleted
            data.rename(columns={'osm_walcycdata_id': 'osm_id'}, inplace=True)
            data['osm_id'] = data['osm_id'].fillna(0)
            Counting.data_to_server(data_to_upload=data, columns_to_upload=CycleCount.column_names(),
                                    table_name='cycle_count_data')
    if params['incident'][0]:
        print('work on incident data')
        local_params = params['incident'][1]
        if local_params['prepare_incident']:
            my_incident = incident_def()
            my_incident.to_file("shp_files/pr_data.gpkg", layer='incident', driver="GPKG")
        if local_params['data_to_server']:
            print('_data_to_server')
            my_incident = gpd.read_file("shp_files/incidents.gpkg", layer='incidents_with_osm_matching')
            # ToDo should be deleted
            my_incident.rename(columns={'osm_walcycdata_id': 'osm_id'}, inplace=True)
            MunichData.data_to_server(my_incident, Incidents.column_names, 'incident_data', engine)

    if params['count_osm'][0]:
        print('count_osm')
        local_params = params['count_osm'][1]
        count_data = gpd.read_file("shp_files/pr_data.gpkg", layer='all_count')
        if local_params['prepare_overlay']:
            print('start overlay')
            osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network')
            osm_data_columns = ['osm_id', 'highway', 'osm_fclass_hir', 'azimuth']
            count_columns = ['index', 'walcycdata_id_cycle', 'walcycdata_id_car', 'cyclecount_count', 'carcount_count',
                             'azimuth']
            results = overlay_count_osm(osm_gdf=osm_data, local_network=count_data, osm_columns=osm_data_columns,
                                        local_columns=count_columns)
            print('finish overlay')
            [item[1].to_file("shp_files/matching_files.gpkg", layer=item[0]) for item in results.items()]
        if local_params['matching']:
            print('_matching')
            osm_munich_overlay = gpd.read_file('shp_files/matching_files.gpkg', layer='overlay')
            map_matching(overlay=osm_munich_overlay, file_count_to_update=count_data, groupby_field='index').to_file(
                "shp_files/matching_files.gpkg", layer='count_osm_matching')
        if local_params['two_ways_matching']:
            print('_two_ways_matching')
            matching_data = gpd.read_file('shp_files/matching_files.gpkg', layer='count_osm_matching')
            my_osm_data = gpd.read_file("shp_files/two_ways.gpkg", layer='osm_gdf')
            my_matching = RoadMatching(matching_data, my_osm_data)
            my_matching.update_by_direction()
            my_matching.osm_matching.to_file("shp_files/two_ways.gpkg", layer='my_matching_dirc', driver="GPKG")
        if local_params['refine_matching']:
            my_matching_data = gpd.read_file("shp_files/two_ways.gpkg", layer='my_matching_dirc', driver="GPKG")
            my_osm_data = gpd.read_file("shp_files/two_ways.gpkg", layer='osm_gdf')
            my_refine_data = RefineMatching(matching_data=my_matching_data, osm_data=my_osm_data)
            my_refine_data.refine()
            my_refine_data.data.to_file("shp_files/matching_files.gpkg", layer='refine_matching', driver="GPKG")
            my_refine_data.pro_pnt_gdf.to_file("shp_files/matching_files.gpkg", layer='refine_matching_pnts',
                                               driver="GPKG")
        if local_params['matching_incidents']:
            my_osm_data = gpd.read_file("shp_files/two_ways.gpkg", layer='osm_gdf')
            my_matching_data = gpd.read_file("shp_files/pr_data.gpkg", layer='incident', driver="GPKG")
            incidents_matchings = IncidentsMatching(local_matching_osm=my_matching_data, osm=my_osm_data)
            osm_data_columns = ['osm_id', 'highway']
            count_columns = ['walcycdata_id']
            results = incidents_matchings.overlay_count_osm(osm_columns=osm_data_columns, local_columns=count_columns)
            print('_finish overlay')
            [item[1].to_file("shp_files/incidents.gpkg", layer=item[0]) for item in results.items()]

            print('_matching_implementation')
            incidents_overlay = gpd.read_file('shp_files/incidents.gpkg', layer='overlay')
            incidents_matchings.map_matching(overlay=incidents_overlay, groupby_field='walcycdata_id')
            incidents_matchings.osm_matching.to_file("shp_files/incidents.gpkg", layer='incidents_with_osm_matching')
        if local_params['matching_to_osm_counting']:
            refine_matching = gpd.read_file("shp_files/matching_files.gpkg", layer='refine_matching', driver="GPKG")
            my_osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network')
            res = OSM.from_local_to_osm(my_osm_data, refine_matching)
            print('_write to disk')
            res.to_file("shp_files/osm/osm.gpkg", layer='openstreetmap_road_network2')
        if local_params['matching_to_osm_incidents']:
            incidents_matching = gpd.read_file("shp_files/incidents.gpkg", layer='incidents_with_osm_matching',
                                               driver="GPKG")
            my_osm_data = gpd.read_file("shp_files//osm/osm.gpkg", layer='openstreetmap_road_network2')
            res = OSM.from_incident_to_osm(osm_network=my_osm_data, local_incidents=incidents_matching)
            print('_write to disk')
            res.to_file("shp_files//osm/osm.gpkg", layer='openstreetmap_road_network_final')
    if params['munich_data'][0]:
        import copy

        print('munich_data')
        local_params = params['munich_data'][1]
        if local_params['prepare_relations']:
            print('_prepare_relations')
            refine_matching = gpd.read_file("shp_files/matching_files.gpkg", layer='refine_matching', driver="GPKG")
            cycle, cars = MunichData.prepare_relations(based_data=refine_matching)
            print('__write results into disk')
            cycle['geometry'] = None,
            cars['geometry'] = None
            # change start_point_id and end_point_id to int
            cycle.to_file("shp_files/munich_data.gpkg", layer='matching_cycle')
            cars.to_file("shp_files/munich_data.gpkg", layer='matching_car')
        if local_params['add_projection_points_to_server']:
            print('add_projection_points_to_server')
            gdf_file = gpd.read_file("shp_files/matching_files.gpkg", layer='refine_matching_pnts', driver="GPKG")
            MunichData.data_to_server(data_to_upload=gdf_file, columns_to_upload=['pnt_id', 'geometry'],
                                      table_name='projection_points', engine=engine, primary_key='pnt_id')
        if local_params['relations_for_server']:
            print('_relations_for_server')
            gpd_files = {'relations_cycles': gpd.read_file("shp_files/munich_data.gpkg", layer='matching_cycle'),
                         'relations_cars': gpd.read_file("shp_files/munich_data.gpkg", layer='matching_car')}
            # ToDo should be deleted for future code
            gpd_files['relations_cycles'] = gpd_files['relations_cycles'][
                gpd_files['relations_cycles']['start_point_id'] != -2]
            gpd_files['relations_cars'] = gpd_files['relations_cars'][
                gpd_files['relations_cars']['start_point_id'] != -2]

            [MunichData.data_to_server(gpd_file[1],
                                       copy.copy(DataForServerDictionaries.COLUMNS),
                                       gpd_file[0], engine,
                                       DataForServerDictionaries.PRIMARY_COLUMNS,
                                       True, False, 'projection_points',
                                       {'start_point_id': 'pnt_id', 'end_point_id': 'pnt_id'}) for gpd_file in
             gpd_files.items()]

    if params['analysis'][0]:
        print('analysis')
        from classes.Analysis import Analysis

        local_params = params['analysis'][1]
        if local_params['osm_network_local_network']:
            cat_file = pd.read_csv('shp_files/cat.csv')
            osm_data = gpd.read_file("shp_files/matching_files.gpkg", layer='osm_gdf')
            matching_data = gpd.read_file("shp_files/matching_files.gpkg", layer='count_osm_matching')[
                'osm_walcycdata_id'].unique()
            results = Analysis.osm_network_local_network(osm_network=osm_data, osm_id_list=matching_data,
                                                         cat_file=cat_file)
            results.to_file("shp_files/matching_files.gpkg", layer='osm_linked_to_counting', driver="GPKG")
        if local_params['analysis_relations']:
            cycle = gpd.read_file("shp_files/munich_data.gpkg", layer='matching_cycle', driver="GPKG")
            cars = gpd.read_file("shp_files/munich_data.gpkg", layer='matching_car', driver="GPKG")
            res, my_dict = Analysis.analysis_relations(cycle_db=cycle, cars_db=cars)
            print(my_dict)
            print(len(res))
            print('_write results into disk')

            res.to_csv(path_or_buf="shp_files/analysis/matching_with_relations.csv")

    if params['data_to_server'][0]:
        local_params = params['data_to_server'][1]

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

        if local_params['combined_network']:
            print("upload combined_network")
            my_car_count = gpd.read_file("shp_files/matching_files.gpkg", layer='count_osm_matching', driver="GPKG")
            my_car_count.to_postgis(name="combined_network", con=engine, schema='production',
                                    if_exists='replace',
                                    dtype={'walcycdata_last_modified': sqlalchemy.types.DateTime})
        if local_params['incidents']:
            pass
