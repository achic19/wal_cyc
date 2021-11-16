from geopandas import GeoDataFrame
import math


class CountOsm:
    def __init__(self):
        self.p = 0

    def overlay_count_osm(self, osm_gdf_0: GeoDataFrame, local_network: GeoDataFrame, osm_columns: list,
                          local_columns: list, dissolve_by: str) -> dict:
        """
        This function calculate between osm entities and local network entities
        :param osm_gdf_0: The OSM network to work on
        :param local_network: count data on local network
        :param local_columns: fields of local network to save while the matching process
        :param osm_columns: fields of osm network to save while the matching process
        :return: list of all the gis files been created during implementation of this stage
        :param dissolve_by: before buffering, the code dissolve osm entities by that field
        """

        # make buffer of 10 meters around each polyline in the both dataframes
        # and calculate the overlay intersection between the two.
        print('osm_dissolve')
        osm_gdf = osm_gdf_0.dissolve(by=dissolve_by).reset_index()

        print('azimuth')
        local_network['azimuth'] = local_network['geometry'].apply(self.calculate_azimuth)
        osm_gdf['azimuth'] = osm_gdf[dissolve_by].apply(azimuth_osm)

        print('osm_buffer')
        osm_buffer = GeoDataFrame(osm_gdf[osm_columns],
                                  geometry=osm_gdf.geometry.buffer(15, cap_style=2), crs=osm_gdf.crs)

        print('count_buffer')
        count_buffer = gpd.GeoDataFrame(local_network[local_columns], crs=local_network.crs,
                                        geometry=local_network.geometry.buffer(15, cap_style=2))

        print('overlay')
        overlay = count_buffer.overlay(osm_buffer, how='intersection')
        # Calculate the percentage field
        print('percentage')
        overlay['areas'] = overlay.area
        count_buffer['areas'] = count_buffer.area
        # The index column in the overlay layer contains the id of the count entity,
        # therefore in order to calculate the rational area between the overlay polygon and the corresponding polygon,
        # the index field becomes the index. In order to save the percentage results,
        # the table is sorted by index and area and then the index is reset.
        overlay.set_index('index', inplace=True)
        temp = (overlay['areas'] * 100 / count_buffer[count_buffer['index'].isin(overlay.index)].set_index('index')[
            'areas']).reset_index()
        temp = temp.sort_values(['index', 'areas'])['areas']
        overlay = overlay.reset_index().sort_values(['index', 'areas'])
        overlay['percentage'] = temp.values
        print('calculate angles between elements')
        overlay['parallel'] = overlay.apply(lambda x: angle_between(x, local_network, osm_gdf), axis=1)
        return {'overlay': overlay, 'osm_buffer': osm_buffer,
                'count_buffer': count_buffer, 'osm_gdf': osm_gdf}

    def calc_azi(self, geom):
        print('calc_azi')
        import math
        return math.degrees(math.atan2(geom[-1]['lon'] - geom[0]['lon'], geom[-1]['lat'] - geom[0]['lat'])) % 360

    def calculate_azimuth(self, row):
        try:
            geo = row.coords
            return math.degrees(math.atan2(geo[-1][0] - geo[0][0], geo[-1][1] - geo[0][1])) % 360
        except:
            return -1
