from sqlalchemy import create_engine, MetaData, text, inspect
import sqlalchemy
import geopandas as gpd
import numpy as np
import pandas as pd
from classes import munich

PRODUCTION_SCHEME = 'production'


def is_table_has_spatial_index(table_name, inspector):
    indice = inspector.get_indexes(table_name, schema=PRODUCTION_SCHEME)
    indice_columns = [indice[i]['column_names'][0] for i in range(len(indice))]
    return 'geometry' in indice_columns


def grant_select_permissions_to_guest(engine):
    with engine.connect() as connection:
        result = connection.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA {PRODUCTION_SCHEME} TO guest"))


def create_wkt_and_index(engine):
    name_table_to_update = ['car_count_data', 'cycle_count_data', 'incident_data', 'openstreetmap_road_network',
                            'projection_points']
    inspector = inspect(engine)
    for table_name in inspector.get_table_names(schema=PRODUCTION_SCHEME):
        if table_name in name_table_to_update:
            table = gpd.GeoDataFrame.from_postgis(f'select * from {PRODUCTION_SCHEME}.{table_name}', engine,
                                                  geom_col='geometry')
            table.to_crs(crs="EPSG:4326", inplace=True)
            if 'wkt' not in table.columns:
                table['wkt'] = table.geometry.astype(str)
                if table_name == 'projection_points':
                    munich.MunichData.data_to_server(table, table.columns.to_list(), table_name, engine,
                                                     primary_key='pnt_id')
                else:
                    munich.MunichData.data_to_server(table, table.columns.to_list(), table_name, engine)
            if not is_table_has_spatial_index(table_name, inspector):
                with engine.connect() as connection:
                    result = connection.execute(
                        text(f"CREATE INDEX ON {PRODUCTION_SCHEME}.{table_name} using gist (geometry)"))


def temp_function():
    table_name = 'projection_points'
    table = gpd.GeoDataFrame.from_postgis(f'select * from {PRODUCTION_SCHEME}.{table_name}', engine,
                                          geom_col='geometry')
    table = table.astype({'pnt_id': float})
    munich.MunichData.data_to_server(table, table.columns.to_list(), table_name, engine, primary_key='pnt_id')


def test_table(table_names):
    for table_name in table_names:
        metadata = MetaData(bind=engine, schema=PRODUCTION_SCHEME)
        my_table = sqlalchemy.Table(table_name, metadata,
                                    autoload=True)
        pass


if __name__ == '__main__':
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')

    # create_wkt_and_index(engine)
    # grant_select_permissions_to_guest(engine)

    print('Done')
    # temp_function()
    test_table(['relations_cycles', 'relations_cars'])
