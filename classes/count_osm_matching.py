from geopandas import GeoDataFrame, GeoSeries
import math
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from classes.munich import *
from classes.osm import OSMAsObject
from networkx.algorithms.shortest_paths import shortest_path
import networkx as nx


class Matching:
    def __init__(self, local_matching_osm: GeoDataFrame, osm: GeoDataFrame):
        """

        :param local_matching_osm:  data on local network
        :param osm: The OSM network to work on
        """
        print("Matching")
        self.osm_matching = local_matching_osm
        self.osm = osm


class RoadMatching(Matching):
    def __init__(self, local_matching_osm: GeoDataFrame, osm: GeoDataFrame):
        super().__init__(local_matching_osm, osm)
        self.osm['pair'] = self.osm['pair'].apply(lambda x: [int(i) for i in x.split(',')])

    def update_by_direction(self):
        print("_update_by_direction")
        self.osm_matching['osm_walcycdata_id'] = self.osm_matching.apply(
            lambda row: self.swap_directions(row[['index', 'azimuth', 'osm_walcycdata_id', 'geometry']]), axis=1)

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
        pair = osm_object['pair'].values[0]
        # no opposite side to the OSM object
        if pair[0] == -1:
            return row['osm_walcycdata_id']
        is_opposite_angle = abs(row.azimuth - osm_object.azimuth.values[0])
        if 30 < is_opposite_angle < 350:
            # Select the opposite direction
            if len(pair) == 1:
                return pair[0]
            else:
                # if there are more than one in the opposite direction select the nearest one
                return self.__find_the_nearest(pair, row)
        else:
            return row['osm_walcycdata_id']

    def __find_the_nearest(self, lines_options, row):
        stringline_list = []
        for edge in lines_options:
            stringline_list.append(self.osm[self.osm['osm_id'] == edge].geometry.values[0])

        # Find the nearest one
        testgdb = GeoDataFrame(geometry=stringline_list)
        res_nearest_ind = testgdb.sindex.nearest(row.geometry, return_distance=True,
                                                 return_all=False)[0][1][0]
        return lines_options[res_nearest_ind]


class IncidentsMatching(Matching):
    def __init__(self, local_matching_osm: GeoDataFrame, osm: GeoDataFrame):
        print('_matching_incidents')
        super().__init__(local_matching_osm, osm)

    def overlay_count_osm(self, osm_columns: list, local_columns: list) -> dict:
        """
        This function calculate between osm entities and local network entities
        :param local_columns: fields of local network to save while the matching process
        :param osm_columns: fields of osm network to save while the matching process
        :return: list of all the gis files been created during implementation of this stage
        """

        # and calculate the overlay intersection between the two.
        print('__osm_buffer')
        osm_buffer = GeoDataFrame(self.osm[osm_columns],
                                  geometry=self.osm.geometry.buffer(15, cap_style=2), crs=MunichData.crs)

        print('__count_buffer')
        count_buffer = gpd.GeoDataFrame(self.osm_matching[local_columns], crs=MunichData.crs,
                                        geometry=self.osm_matching.geometry.buffer(15))

        print('__overlay')
        overlay = count_buffer.overlay(osm_buffer, how='intersection')
        # Calculate the percentage field
        overlay['areas'] = overlay.area
        count_buffer['areas'] = count_buffer.area
        # The index column in the overlay layer contains the id of the count entity,
        # therefore in order to calculate the rational area between the overlay polygon and the corresponding polygon,
        # the index field becomes the index. In order to save the percentage results,
        # the table is sorted by index and area and then the index is reset.
        return {'overlay': overlay, 'osm_buffer': osm_buffer,
                'count_buffer': count_buffer}

    def map_matching(self, overlay: GeoDataFrame, groupby_field: str):
        """
        map between the local network to the osm network based on the largest overlay between two entities
        :param overlay: polygons of overlay
        :param groupby_field:
        :return:
        """
        print('_start map matching')
        print('__find the best matching')
        # for each count object return the osm id with the larger overlay with him
        matching_info = overlay.groupby(groupby_field).apply(IncidentsMatching.__find_optimal_applicant)
        print('__finish the best matching')
        matching_info = matching_info[matching_info.notna()]
        # update cycle count data with the corresponding osm id
        matching_info.sort_index(inplace=True)

        self.osm_matching.set_index(groupby_field, drop=False, inplace=True)
        self.osm_matching.sort_index(inplace=True)
        self.osm_matching['osm_walcycdata_id'] = -1
        self.osm_matching['osm_walcycdata_id'][self.osm_matching[groupby_field].isin(matching_info[groupby_field])] = \
            matching_info['osm_id']
        self.osm_matching.reset_index(drop=True, inplace=True)
        print('_finish map matching')

    @staticmethod
    def __find_optimal_applicant(group):
        """
        By analyzing several conditions, this function finds the best candidates out of all the options:
        1. The highest in the hierarchy should be the first ti check
        2. Then these with the largest overlay should be chosen
        :param group: group of applicants (osm entities)
        :return:
        """
        if len(group) > 1:
            group.sort_values(['areas'], ascending=False, inplace=True)
            # the code prefers not to match to footway or steps or path
            for row_temp in group.iterrows():
                row = row_temp[1]
                if (row['highway'] != 'footway') & (row['highway'] != 'steps') & (row['highway'] != 'path'):
                    return row
            return group.iloc[0]
        return group.iloc[0]


class RefineMatching(MunichData):
    def __init__(self, matching_data: GeoDataFrame,
                 osm_data: GeoDataFrame):
        print('create RefineMatching object')
        self.data = matching_data
        self.osm = osm_data.set_index('osm_id')

        # The following parameters will be used to populate the projection points list
        self.pro_list = []
        self.cur_pnt_inx = 0
        self.pro_pnt_gdf = GeoDataFrame()

        # data to match for than  one osm object
        self.osm_as_graph = OSMAsObject.create_osm_obj_from_local_machine()
        self.osm_as_graph.gdp_pnt.set_index('id', inplace=True, drop=True)
        # Objects related to the matching
        self.ind_pnt = int
        self.list_of_osm_obj = []
        self.osm_obj = GeoSeries()

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
                return -1, -1, -1
            osm_object = self.osm.loc[row.osm_walcycdata_id]
            osm_az = osm_object.azimuth
            if osm_az == -1:
                # the matching osm object has circular feature
                return -2, -2, -2
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
                return -4, -4, -4
            try:
                self.osm_obj = osm_object
                # find more osm
                self.__project_pnt_osm_obj(pnts_list[0], 0)
                self.list_of_osm_obj.reverse()
                osm_ids = self.list_of_osm_obj
                first_proj = self.ind_pnt
                self.osm_obj = osm_object
                self.__project_pnt_osm_obj(pnts_list[-1], -1)
                lasts_osm = self.list_of_osm_obj
                last_proj = self.ind_pnt

                # append two lists into one
                osm_ids.extend(lasts_osm)
                osm_ids.remove(osm_object.name)

                return first_proj, osm_ids, last_proj
            except NotImplementedError:
                # ToDo for multilinestring which will not be appear in the new osm data
                return -4, -4, -4

        new_fields = ['start_point_id', 'osm_ids', 'end_point_id']
        print('_refine')
        self.data[new_fields] = list(self.data.apply(find_finer_data_osm, axis=1))
        # Update gdf with the new points
        self.pro_pnt_gdf['geometry'] = self.pro_list
        self.pro_pnt_gdf.reset_index(inplace=True)
        self.pro_pnt_gdf.rename(columns={'index': 'pnt_id'}, inplace=True)
        self.pro_pnt_gdf.crs = MunichData.crs

        self.data['osm_ids'] = self.data['osm_ids'].apply(
            lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x))

    def __project_pnt_osm_obj(self, pnt: tuple, ind: int):
        """
        Project the point onto the osm object.
        More work is required if the projection points are the ends of OSM objects.
        :param pnt: pnt to project
        :param pnt_loc:
        string that retain information about the point location on the local object (first or last)
        :param ind:pnt index in pnt list of local line object
        :return:
        """
        # osm_pnt store the first or last point of osm string line object
        osm_geometry = self.osm_obj.geometry
        osm_coordinates = list(osm_geometry.coords)
        osm_pnt = osm_coordinates[ind]
        local_pnt_shpy = Point(pnt)
        osm_pnt_shpy = Point(osm_pnt)

        self.list_of_osm_obj = [self.osm_obj.name]
        proj = RefineMatching.__make_projection(osm_geometry, local_pnt_shpy)
        # The projection might be executed against a different OSM object
        # if the projection point equals osm_pnt and the distance is more than 100 meters
        if round(proj[0], 0) == round(osm_pnt[0], 0) and round(proj[1], 0) == round(
                osm_pnt[1], 0) and ((pnt[0] - osm_pnt[0]) ** 2 + (pnt[1] - osm_pnt[1]) ** 2) ** 0.5 > 50:
            self.__long_local_object(local_pnt_shpy, osm_pnt_shpy, proj)
        else:
            self.__update_points_list(Point(proj))

    def __long_local_object(self, local_pnt_shpy: Point, osm_pnt_shpy: Point, proj: list):
        """
        If the projection point is on the OSM end lines and the distance between the local point and
        the OSM ends is greater than 100 meters, more work needs to be done.
        :param local_pnt_shpy:
        :param osm_pnt_shpy:
        :param pnt_loc:
        :return:
        """

        # find the nearest node on the graph to the local_pnt

        pnt_osm_in_graph = self.__find_point_name_on_the_graph(osm_pnt_shpy)
        try:
            next_lines_options = self.osm_as_graph.graph.edges(pnt_osm_in_graph)
        except nx.exception.NetworkXError:
            # in case of point not on the graph, it should be deleted and than other point will be
            print('point {} not not in graph therefore will be deleted'.format(pnt_osm_in_graph))
            self.osm_as_graph.gdp_pnt.drop(self.osm_as_graph.gdp_pnt.loc[pnt_osm_in_graph].name, inplace=True)
            pnt_osm_in_graph = self.__find_point_name_on_the_graph(osm_pnt_shpy)
            next_lines_options = self.osm_as_graph.graph.edges(pnt_osm_in_graph)

        # Restore the geometries of the examined stringline
        stringline_list = []
        for edge in next_lines_options:
            osm_id_obj = self.osm_as_graph.graph.edges[edge[0], edge[1]]['osmid']
            stringline_list.append(self.osm.loc[osm_id_obj])

        # Find the nearest one
        testgdb = GeoDataFrame(stringline_list)
        res_nearest_ind = testgdb.sindex.nearest([local_pnt_shpy], return_distance=True,
                                                 return_all=False)[0][1][0]
        res_nearest_osm_id = stringline_list[res_nearest_ind].name
        # if it is still same points:
        if res_nearest_osm_id == self.osm_obj.name:
            self.__update_points_list(Point(proj))
            return
        # the code continues to check more relevant osm segments to the current local segment
        self.list_of_osm_obj.append(res_nearest_osm_id)
        self.osm_obj = self.osm.loc[res_nearest_osm_id]
        self.__find_more_edges(local_pnt_shpy)

    def __find_more_edges(self, local_pnt_shpy):

        proj = RefineMatching.__make_projection(self.osm_obj.geometry, local_pnt_shpy)
        pnt = list(local_pnt_shpy.coords)[0]
        # Which point to check
        osm_coordinates = list(self.osm_obj.geometry.coords)
        osm_pnt_0 = Point(osm_coordinates[0])
        osm_pnt_1 = Point(osm_coordinates[-1])
        osm_pnt_shpy = osm_pnt_0 if osm_pnt_0.distance(local_pnt_shpy) < osm_pnt_1.distance(
            local_pnt_shpy) else osm_pnt_1
        osm_pnt = list(osm_pnt_shpy.coords)[0]
        if round(proj[0], 0) == round(osm_pnt[0], 0) and round(proj[1], 0) == round(
                osm_pnt[1], 0) and ((pnt[0] - osm_pnt[0]) ** 2 + (pnt[1] - osm_pnt[1]) ** 2) ** 0.5 > 50:
            self.__long_local_object(local_pnt_shpy, osm_pnt_shpy, proj)
        else:
            self.__update_points_list(Point(proj))

    def __find_point_name_on_the_graph(self, osm_pnt_shpy):
        pnts_on_graph = self.osm_as_graph.gdp_pnt.sindex.nearest([osm_pnt_shpy],
                                                                 return_distance=True, return_all=False)
        osm_inx = pnts_on_graph[0][1][0]
        return self.osm_as_graph.gdp_pnt.iloc[osm_inx].name

    def __update_points_list(self, proj: Point):
        """
        Updates the projection list with a new projection point and returns its index
        :param proj:
        :return:
        """
        self.ind_pnt = self.cur_pnt_inx
        self.pro_list.append(proj)
        self.cur_pnt_inx += 1

    @staticmethod
    def __make_projection(line: LineString, point: Point) -> list:
        return list(line.interpolate(line.project(point)).coords)[0]


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
