from classes.munich import *


class Incidents(MunichData):
    # Data

    column_names = ['walcycdata_id', 'incident_id', 'incident_land',
                    'incident_district', 'incident_region', 'incident_year',
                    'incident_month', 'incident_hour', 'incident_week_day',
                    'incident_personal_injury', 'incident_class', 'incident_type',
                    'incident_light_condition', 'incident_with_bycycle',
                    'incident_with_passenger_car', 'incident_with_passenger',
                    'incident_with_motorcyle', 'incident_with_goods', 'incident_with_other',
                    'incident_ref_x', 'incident_ref_y', 'incident_gcswgs84_x',
                    'incident_gcswgs84_y', 'incident_surface_condition',
                    'walcycdata_is_valid', 'walcycdata_last_modified', 'osm_id',
                    'geometry']
