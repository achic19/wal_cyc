import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def incident_def():
    # upload files
    if params['test']:
        incident = gpd.read_file('shp_files/Incident_data.shp', rows=10)
    else:
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
        'osm_id'] = pd.Series(map(lambda x: '1' + x, np.arange(n).astype(str))), True, date, ''
    return incident


def cycle_count_def():
    # upload files
    if params['test']:
        cycle_count = gpd.read_file('shp_files/cycle_count_data.shp', rows=10)
    else:
        cycle_count = gpd.read_file('shp_files/cycle_count_data.shp')

    # change table names
    cycle_count.rename(
        columns={"NO": "Cyclecount_id", "ZAEHLUN~19": "Cyclecount_count"}, inplace=True)

    # drop fields
    cycle_count = cycle_count[['Cyclecount_id', 'Cyclecount_count', 'geometry']]

    # new fields, walcycdata_id = a unique number with leading 2 for cycle count
    n = len(cycle_count)
    cycle_count['walcycdata_id'], cycle_count['walcycdata_is_valid'], cycle_count['walcycdata_last_modified'], \
    cycle_count['osm_id'], cycle_count['Cyclecount_timestamp_start'], cycle_count[
        'Cyclecount_timestamp_end'] = pd.Series(
        map(lambda x: '2' + x, np.arange(n).astype(str))), True, date, '', '', ''

    return cycle_count.dropna()


def car_count_def():
    # upload files
    if params['test']:
        car_count = gpd.read_file('shp_files/cars_count_data.shp', rows=10)
    else:
        car_count = gpd.read_file('shp_files/cars_count_data.shp')

    # change table names
    car_count.rename(
        columns={"NO": "Carcount_id", "ZOhlung_~7": "Carcount_count"}, inplace=True)

    # drop fields
    car_count = car_count[['Carcount_id', 'Carcount_count', 'geometry']]

    # new fields, walcycdata_id = a unique number with leading 3 for cycle count
    n = len(car_count)
    car_count['walcycdata_id'], car_count['walcycdata_is_valid'], car_count['walcycdata_last_modified'], \
        car_count['osm_id'], car_count['Carcount_timestamp_start'], car_count[
        'Carcount_timestamp_end'] = pd.Series(
        map(lambda x: '3' + x, np.arange(n).astype(str))), True, date, '', '', ''

    return car_count.dropna()


def osm_file_def():
    if params['test']:
        osm_file = gpd.read_file('shp_files/OpenStreetMap_ road_network.shp', rows=10)

    else:
        osm_file = gpd.read_file('shp_files/OpenStreetMap_ road_network.shp')

    # drop duplicate
    osm_file.drop_duplicates(subset=['osm_id'], inplace=True)
    # change table names
    osm_file.rename(
        columns={"fclass": "osm_fclass", "name": "osm_name", "oneway": "osm_oneway", "maxspeed": "osm_maxspeed",
                 "layer": "osm_layer", "bridge": "osm_bridge", "tunnel": "osm_tunnel"}, inplace=True)

    # drop fields
    osm_file.drop(
        columns=['cat', 'code', 'ref', 'length', 'length_lix', 'index', 'all_densit', 'weight_den'], inplace=True)

    # new fields, walcycdata_id = a unique number with leading 4 for osm
    n = len(osm_file)
    osm_file['walcycdata_id'], osm_file['walcycdata_is_valid'], osm_file['walcycdata_last_modified'] = pd.Series(
        map(lambda x: '4' + x, np.arange(
            n).astype(str))), True, date

    return osm_file


if __name__ == '__main__':
    params = {'incident': False, 'cycle_count': False, 'car_count': True, 'osm_file': True, 'test': False}

    date = pd.to_datetime("today")
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')

    if params['incident']:
        print('create incident table')
        my_incident = incident_def()
        if params['test']:
            print('incident csv table')
            my_incident.set_index('walcycdata_id').to_csv('shp_files/new_shp_files/incident_data.csv')
        else:
            print('upload incident table')
            my_incident.to_postgis(name="incident_data", con=engine, schema='production', if_exists='replace')

    if params['cycle_count']:
        print('create cycle_count table')
        my_cycle_count = cycle_count_def()
        if params['test']:
            print('csv cycle_count table')
            my_cycle_count.set_index('walcycdata_id').to_csv('shp_files/new_shp_files/cycle_count_data.csv')
        else:
            print('upload cycle_count table')
            my_cycle_count.to_postgis(name="cycle_count_data", con=engine, schema='production', if_exists='replace')

    if params['car_count']:
        print('create car_count table')
        my_car_count = car_count_def()
        if params['test']:
            print('csv car_count table')
            my_car_count.set_index('walcycdata_id').to_csv('shp_files/new_shp_files/car_count_data.csv')
        else:
            print('upload cycle_count table')
            my_car_count.to_postgis(name="car_count_data", con=engine, schema='production', if_exists='replace')
    if params['osm_file']:
        print('create osm table')
        my_osm_file = osm_file_def()
        if params['test']:
            print('osm csv table')
            my_osm_file.set_index('walcycdata_id').to_csv('shp_files/new_shp_files/openStreetMap_road_network.csv')
        else:
            print('upload osm table')
            my_osm_file.to_postgis(name="openstreetmap_road_network", con=engine, schema='production',
                                   if_exists='replace')
