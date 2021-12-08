from geopandas import GeoDataFrame
import math
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from classes.munich import *
from classes.osm import OSMAsObject
from networkx.algorithms.shortest_paths import shortest_path


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

            def project_pnt_osm_obj(pnt: tuple, pnt_loc: str, ind: int):
                """
                Project the point onto the osm object.
                More work is required if the projection points are the ends of OSM objects.
                :param pnt: pnt to project
                :param pnt_loc:
                string that retain information about the point location on the local object (first or last)
                :param ind:pnt index in pnt list of local line object
                :return:
                """
                osm_pnt = osm_coordinates[ind]
                local_pnt_shpy = Point(pnt)
                osm_pnt_shpy = Point(osm_pnt)
                proj = list(osm_object.geometry.interpolate(osm_geometry.project(local_pnt_shpy)).coords)[0]

                if round(proj[0], 0) == round(osm_pnt[0], 0) and round(proj[1], 0) == round(
                        osm_pnt[1], 0) and ((pnt[0] - osm_pnt[0]) ** 2 + (pnt[1] - osm_pnt[1]) ** 2) ** 0.5 > 100:
                    # If the projection point is on the OSM end lines and the distance between the local point and
                    # the OSM ends is greater than 50 meters, more work needs to be done.
                    # find the nearest node on the graph to teh local_pnt
                    pnts_on_graph = self.osm_as_graph.gdp_pnt.sindex.nearest([local_pnt_shpy, osm_pnt_shpy],
                                                                             return_distance=True, return_all=False)
                    indices_on_tree = pnts_on_graph[0][1]
                    local_inx = indices_on_tree[0]
                    osm_inx = indices_on_tree[1]
                    if local_inx == osm_inx:
                        # if the local_inx (source) equal to osm_inx (target), return osm point
                        proj_index = self.cur_pnt_inx
                        self.pro_list.append(osm_pnt_shpy)
                        self.cur_pnt_inx += 1
                        return proj_index
                        # Find the route between the route to target
                    if pnt_loc == 'first':
                        source = self.osm_as_graph.gdp_pnt.iloc[local_inx].name
                        des = self.osm_as_graph.gdp_pnt.iloc[osm_inx].name
                    else:
                        source = self.osm_as_graph.gdp_pnt.iloc[osm_inx].name
                        des = self.osm_as_graph.gdp_pnt.iloc[local_inx].name
                    # Calculate the shortest path
                    shortest = shortest_path(self.osm_as_graph.graph, source=source, target=des, weight='length')
                    # Go over all the shortest points to find all the osm objects matching the local object
                    k = 0
                    list_of_osm_obj = []
                    while k < len(shortest) - 1:
                        osm_ogj = self.osm_as_graph.graph.edges[shortest[k], shortest[k + 1], 0]['osmid']
                        if isinstance(osm_ogj, list):
                            list_of_osm_obj.append(osm_ogj[0])
                        else:
                            list_of_osm_obj.append(osm_ogj)
                        k = k + 1
                    return shortest

                else:
                    proj_index = self.cur_pnt_inx
                    self.pro_list.append(Point(proj))
                    self.cur_pnt_inx += 1
                    return proj_index

            try:
                osm_geometry = osm_object.geometry
                osm_coordinates = list(osm_geometry.coords)
                first_proj = project_pnt_osm_obj(pnts_list[0], 'first', 0)
                last_proj = project_pnt_osm_obj(pnts_list[-1], 'last', -1)
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
