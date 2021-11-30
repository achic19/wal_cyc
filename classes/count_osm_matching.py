from geopandas import GeoDataFrame
import math
import geopandas as gpd
import numpy as np


class Matching:
    def __init__(self, osm_matching: GeoDataFrame, osm):
        print("Matching")
        self.osm_matching = osm_matching
        self.osm = osm

    def update_by_direction(self):
        print("_update_by_direction")
        self.osm_matching['osm_walcycdata_id'] = self.osm_matching.apply(
            lambda row: self.swap_directions(row[['azimuth', 'osm_walcycdata_id']]), axis=1)
        pass

    def swap_directions(self, row):
        """
        This method check wether the opposite side of road should be linked into that matching object
        :param row:
        :return:
        """
        # no matching object
        if row['osm_walcycdata_id'] == -1:
            return -1
        osm_object = self.osm[self.osm['osm_id'] == row['osm_walcycdata_id']]
        # no opposite side to the OSM object
        if osm_object['pair'].isnull().any():
            return -1
        if abs(row.azimuth - osm_object.azimuth.values[0]) > 30:
            # Select the opposite direction
            return osm_object['pair'].values[0]
        else:
            return row['osm_walcycdata_id']


if __name__ == '__main__':
    import os

    os.chdir(r'D:\Users\Technion\Sagi Dalyot - AchituvCohen\WalCycData\shared_project')
    matching_data = gpd.read_file('shp_files/matching_files.gpkg', layer='count_osm_matching')
    my_osm_data = gpd.read_file("shp_files/two_ways.gpkg", layer='osm_gdf')
    my_matching = Matching(matching_data, my_osm_data)
    my_matching.update_by_direction()
    print("_write to disk")
    my_matching.osm_matching.to_file("shp_files/two_ways.gpkg", layer='my_matching_dirc', driver="GPKG")
