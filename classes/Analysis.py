from geopandas import GeoDataFrame, GeoSeries


class Analysis:
    @staticmethod
    def osm_network_local_network(osm_network: GeoDataFrame, osm_id_list: list, cat_file: GeoDataFrame):
        """
        This method calculate f
        :param osm_network:
        :param osm_id_list:
        :param cat_file:
        :return:
        """
        osm_to_local = osm_network[osm_network['osm_id'].isin(osm_id_list)]
        cat_file.set_index('categories', inplace=True)
        osm_to_local["osm_fclass_hir"] = cat_file.loc[osm_to_local["highway"]]['hierarchical values'].values
        return osm_to_local
