import geopandas as gpd
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import networkx as nx
import osmnx as ox


def create_relations_incidents(network_df, incidents_df):
    res = network_df.sjoin_nearest(incidents_df, how='inner', max_distance=None, lsuffix='left', rsuffix='right',
                                   distance_col=None)
    res.to_file('fff.shp')


def my_from_postgis(sql):
    return gpd.GeoDataFrame.from_postgis(sql, engine)


if __name__ == '__main__':
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')
    network = my_from_postgis("SELECT * FROM production/openstreetmap_road_network")
    incidents = my_from_postgis("SELECT * FROM production/incident_data")
    create_relations_incidents()
