import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

if __name__ == '__main__':
    # upload files
    osm_file = gpd.read_file('shp_files/OpenStreetMap_ road_network.shp')
    incident = gpd.read_file('shp_files/Incident_data.shp')
    cycle_count = gpd.read_file('shp_files/cycle_count_data.shp')

    # change table names

    osm_file.rename(
        columns={"fclass": "osm_fclass", "name": "osm_name", "oneway": "osm_oneway", "maxspeed": "osm_maxspeed",
                 "layer": "osm_layer", "bridge": "osm_bridge", "tunnel": "osm_tunnel"}, inplace=True)
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

    cycle_count.rename(
        columns={"NO": "Cyclecount_id", "ZAEHLUN~19": "Cyclecount_count"}, inplace=True)

    # drop fields
    osm_file.drop(
        columns=['cat', 'code', 'ref', 'length', 'length_lix', 'index', 'all_densit', 'weight_den'], inplace=True)
    incident.drop(
        columns=['Municipali', 'severity'], inplace=True)
    cycle_count = cycle_count[['Cyclecount_id', 'Cyclecount_count', 'geometry']]

    # new fields
    date = pd.to_datetime("today")
    n1 = len(incident)
    n2 = n1 + len(osm_file)
    n3 = n2 + len(cycle_count)
    incident['walcycdata_id'], incident['walcycdata_is_valid'], incident['walcycdata_last_modified'], incident[
        'osm_id'] = np.arange(n1), True, date, ''
    osm_file['walcycdata_id'], osm_file['walcycdata_is_valid'], osm_file['walcycdata_last_modified'] = np.arange(n1,
                                                                                                                 n2), \
                                                                                                       True, date
    cycle_count['walcycdata_id'], cycle_count['walcycdata_is_valid'], cycle_count['walcycdata_last_modified'], \
    cycle_count['osm_id'], cycle_count['Cyclecount_timestamp_start'], cycle_count[
        'Cyclecount_timestamp_end'] = np.arange(n2, n3), True, date, '', '', ''

    cycle_count = cycle_count.dropna()
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')

    osm_file.to_postgis(name="OpenStreetMap_road_network", con=engine, schema='production')
    incident.to_postgis(name="incident_data", con=engine, schema='production')
    cycle_count.to_postgis(name="cycle_count_data", con=engine, schema='production')

    # export new tables
    # osm_file.set_index('walcycdata_id').to_csv('shp_files/new_shp_files/OpenStreetMap_road_network.csv')
    # incident.set_index('walcycdata_id').to_csv('shp_files/new_shp_files/incident_data.csv')
    # cycle_count.set_index('walcycdata_id').to_csv('shp_files/new_shp_files/cycle_count_data.csv')
