from geopandas import GeoDataFrame, GeoSeries
import pandas as pd


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

    @staticmethod
    def analysis_relations(cycle_db: GeoDataFrame, cars_db: GeoDataFrame) -> tuple[pd.DataFrame, dict]:
        """
        The process starts by merging two geo dataframes (without geometry) and then finds the relationships between the
        local and osm networks among the optional relations in the dict object
        :param cycle_db:
        :param cars_db:
        :return:
        """
        print('_analysis relations')
        dic = {'one_to_one': 0, 'many_to_one': 0, 'one_to_many': 0, 'many_to_many': 0}
        relation = []
        print('__merge')
        db = cycle_db.merge(cars_db, how='outer', on=['index', 'osm_id'])

        def find_relation_for_row(row):
            # if len(row) ==1
            # _ if len(group2[row['osm_id']==1
            # __ the relation is one to one
            # _ else
            # __ the relation is many to one
            # else:
            # _ for each relation:
            # __ if len(group2[row['osm_id']==1
            # ___ the relation is one to many
            # __ else
            # ___ the relation is many to many
            def iterate_over(osm_id, relation_names):
                if len(group_2.get_group(osm_id)) == 1:
                    dic[relation_names[0]] += 1
                    relation.append(relation_names[0])
                else:
                    dic[relation_names[1]] += 1
                    relation.append(relation_names[1])

            if len(row) == 1:
                iterate_over(row.iloc[0]['osm_id'], list(dic.keys())[:2])
            else:
                row.apply(lambda x: iterate_over(x['osm_id'], list(dic.keys())[2:]), axis=1)

        print('__find relations')
        group_1 = db.groupby('index')
        group_2 = db.groupby('osm_id')
        group_1.apply(find_relation_for_row)
        db['relation'] = relation
        return db, dic
