import geopandas as gpd
from geopandas import GeoDataFrame
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import networkx as nx
import osmnx as ox


def create_relations_incidents(network_df: GeoDataFrame, incidents_df: GeoDataFrame):

    res = network_df.sjoin_nearest(incidents_df, how='inner', max_distance=None, lsuffix='left', rsuffix='right',
                                   distance_col=None)
    res.to_file('fff.shp')


# def my_from_postgis(sql):
#     return gpd.GeoDataFrame.from_postgis(sql, engine)


if __name__ == '__main__':
    # engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')
    # network = my_from_postgis("SELECT * FROM production/openstreetmap_road_network")
    # incidents = my_from_postgis("SELECT * FROM production/incident_data")
    network = gpd.read_file("shp_files/data_from_server/osm_net.shp")
    incidents = gpd.read_file("shp_files/data_from_server/incs.shp")
    create_relations_incidents(network, incidents)

# import geopandas as gpd
#
# import psycopg2  # (if it is postgres/postgis)
#
# con = psycopg2.connect(database="your database", user="user", password="password",
#     host="your host")
#
# sql = "select geom, x,y,z from your_table"
#
# df = gpd.GeoDataFrame.from_postgis(sql, con, geom_col='geom' )