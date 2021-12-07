from geopandas import GeoDataFrame
import math
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from classes.munich import *
from classes.osm import OSMAsObject

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


class RefineMatching(MunichData):
    def __init__(self, matching_data: GeoDataFrame,
                 osm_data: GeoDataFrame):
        self.data = matching_data
        self.osm = osm_data.set_index('osm_id')

        # The following parameters will be used to populate the projection points list
        self.pro_list = []
        self.cur_pnt_inx = 0
        self.pro_pnt_gdf = GeoDataFrame()

        # data to match for than  one osm object
        self.osm_as_graph = OSMAsObject.create_osm_obj_from_local_machine()

    def refine(self):
        """
        The method check in more detail the matching and might add more information
        :return:
        """

        def find_finer_data_osm(row):
            """
            The method check in more detail the matching for a specific object and might add more information
            :param row: the local object to be examined
            :return:
            """
            if row.osm_walcycdata_id == -1:
                # no matching osm object
                return -1, -1, -1, -1
            osm_object = self.osm.loc[row.osm_walcycdata_id]
            osm_az = osm_object.azimuth
            if osm_az == -1:
                # the matching osm object has circular feature
                return -2, -2, -2, -2
            # First, if necessary, reverse the local object to be in the same direction as the osm object
            angle = abs(row.azimuth - osm_az)
            try:
                if 90 < angle < 270:
                    pnts_list = list(row.geometry.coords)
                    pnts_list.reverse()
                else:
                    pnts_list = list(row.geometry.coords)
            except NotImplementedError:
                # ToDo for multilinestring which will not be appear in the new local network data
                return -4, -4, -4, -4

            def project_pnt_osm_obj(pnt: tuple, ind: int):
                """
                Project the point onto the osm object.
                More work is required if the projection points are the ends of OSM objects.
                :param pnt: pnt to project
                :param ind:pnt index in pnt list of local line object
                :return:
                """

                proj = list(osm_object.geometry.interpolate(osm_geometry.project(Point(pnt))).coords)[0]
                osm_pnt = osm_coordinates[ind]
                if round(proj[0], 0) == round(osm_pnt[0], 0) and round(proj[1], 0) == round(
                        osm_pnt[1], 0) and ((pnt[0] - osm_pnt[0]) ** 2 + (pnt[1] - osm_pnt[1]) ** 2) ** 0.5 > 50:
                    # If the projection point is on the OSM end lines and the distance between the local point and
                    # the OSM ends is greater than 50 meters, more work needs to be done.
                    return -3
                else:
                    proj_index = self.cur_pnt_inx
                    self.pro_list.append(Point(proj))
                    self.cur_pnt_inx += 1
                    return proj_index

            try:
                osm_geometry = osm_object.geometry
                osm_coordinates = list(osm_geometry.coords)
                first_proj = project_pnt_osm_obj(pnts_list[0], 0)
                last_proj = project_pnt_osm_obj(pnts_list[-1], -1)
                return row.osm_walcycdata_id, first_proj, row.osm_walcycdata_id, last_proj
            except NotImplementedError:
                # ToDo for multilinestring which will not be appear in the new osm data
                return -4, -4, -4, -4

        new_fields = ['start_osm_id', 'start_point_id', 'end_osm_id', 'end_point_id']
        print('refine')
        self.data[new_fields] = list(self.data.apply(find_finer_data_osm, axis=1))
        # Update gdf with the new points
        self.pro_pnt_gdf['geometry'] = self.pro_list
        self.pro_pnt_gdf.reset_index(inplace=True)
        self.pro_pnt_gdf.rename(columns={'index': 'pnt_id'}, inplace=True)
        self.pro_pnt_gdf.crs = MunichData.crs


if __name__ == '__main__':
    import os

    os.chdir(r'D:\Users\Technion\Sagi Dalyot - AchituvCohen\WalCycData\shared_project')
    my_matching_data = gpd.read_file("shp_files/matching_files.gpkg", layer='count_osm_matching', driver="GPKG")
    my_osm_data = gpd.read_file("shp_files/pr_data.gpkg", layer='openstreetmap_road_network')
    my_refine_data = RefineMatching(matching_data=my_matching_data, osm_data=my_osm_data)
    my_refine_data.refine()
    my_refine_data.data.to_file("shp_files/matching_files.gpkg", layer='refine_matching', driver="GPKG")
    my_refine_data.pro_pnt_gdf.to_file("shp_files/matching_files.gpkg", layer='refine_matching_pnts', driver="GPKG")
    # matching_data = gpd.read_file('shp_files/matching_files.gpkg', layer='count_osm_matching')
    # my_osm_data = gpd.read_file("shp_files/two_ways.gpkg", layer='osm_gdf')
    # my_matching = Matching(matching_data, my_osm_data)
    # my_matching.update_by_direction()
    # print("_write to disk")
    # my_matching.osm_matching.to_file("shp_files/two_ways.gpkg", layer='my_matching_dirc', driver="GPKG")
