from sqlalchemy import create_engine, MetaData
import sqlalchemy
from sqlalchemy.engine.base import Engine
from migrate.changeset.constraint import PrimaryKeyConstraint
from geopandas import GeoDataFrame
import pandas as pd
import geopandas as gpd
from typing import Tuple


class MunichData:
    schema = 'production'
    crs = "EPSG:3857"
    date = pd.to_datetime("today")

    @staticmethod
    def from_str_to_list_of_str(db: GeoDataFrame, field_name: str):
        """
        get a panda column as string and return it as a list of ints
        :param db:
        :param field_name:
        :return:
        """
        return db[field_name].apply(lambda x: [int(i) for i in x.split(',')])

    @staticmethod
    def clip_data(file: GeoDataFrame, clip_file: GeoDataFrame) -> GeoDataFrame:
        # "Sjoin" is used to remove the entire segment that intersects the clip.
        clipped = file.to_crs(MunichData.crs).sjoin(clip_file, how="inner", predicate='within')
        return clipped[~clipped.is_empty][file.columns]

    @staticmethod
    def prepare_relations(based_data: GeoDataFrame) -> Tuple[GeoDataFrame, GeoDataFrame]:
        """
        Create two new table that keep only the cycle_id and car_id with the matching osm
        :param based_data:
        :return:
        """
        print('__preprocessing tasks')
        # fill null
        based_data[['walcycdata_id_car', 'walcycdata_id_cycle']] = based_data[
            ['walcycdata_id_car', 'walcycdata_id_cycle']].fillna(-1)
        based_data[['carcount_count', 'cyclecount_count']] = based_data[['carcount_count', 'cyclecount_count']].fillna(
            0)
        # for osm_ids: from string to list of ints
        based_data['osm_ids'] = MunichData.from_str_to_list_of_str(db=based_data,
                                                                   field_name='osm_ids')
        # remove the -1
        based_data = based_data[based_data['osm_walcycdata_id'] != -1]

        def combine_network_to_one_user(db: GeoDataFrame, field_name: str) -> DataForServerDictionaries:
            """
            update the data_dictionary with data from db (cars or cycles data with counting)
            :param db:
            :param field_name:
            :return:
            """

            # create new dictionary
            print('__create new dictionaries of DataForServerDictionaries class')
            data_dictionary = DataForServerDictionaries()
            # for each row update the relevant dictionary
            db.apply(
                lambda x: data_dictionary.find_relevant_data_to_update_dictionary(x, field_name),
                axis=1)
            return data_dictionary

        print('__work with cycle data')
        # For the new table, store only rows with cycles or cars data counts
        data_dictionary_cycle = combine_network_to_one_user(based_data[based_data['cyclecount_count'] != 0],
                                                            'walcycdata_id_cycle')
        print('__work with car data')
        data_dictionary_car = combine_network_to_one_user(based_data[based_data['carcount_count'] != 0],
                                                          'walcycdata_id_car')
        # Make the new dictionaries into a databases
        print('__make the new dictionaries into a databases')
        return GeoDataFrame(data_dictionary_cycle.dict), GeoDataFrame(data_dictionary_car.dict)

    @staticmethod
    def data_to_server(data_to_upload: GeoDataFrame,
                       columns_to_upload: list,
                       table_name: str,
                       engine: Engine,
                       primary_key='walcycdata_id'):
        """
        Upload GeoDataFrame  into schema in engine
        and only with the columns specified in the columns_to_upload parameter
        :param table_name:
        :param data_to_upload:
        :param columns_to_upload:
        :param engine:
        :param primary_key: for the original table, 'walcycdata_id' field is the primary one.
        For the new table the primary key is different
        :return:
        """

        # work only with data in columns_to_upload
        print('_work only with data in columns_to_upload')
        data_to_upload = data_to_upload[columns_to_upload]

        # update the last time data were changed
        print('_update the last time data were changed')
        data_to_upload['walcycdata_last_modified'] = MunichData.date
        # upload data
        print('_upload data')
        if type(data_to_upload) is GeoDataFrame:
            data_to_upload.to_postgis(name=table_name, con=engine, schema=MunichData.schema,
                                      if_exists='replace',
                                      dtype={'walcycdata_last_modified': sqlalchemy.types.DateTime})
        else:
            data_to_upload.to_sql(name=table_name, con=engine, schema=MunichData.schema,
                                  if_exists='replace',
                                  dtype={'walcycdata_last_modified': sqlalchemy.types.DateTime})

        # define primary key
        print('_define primary key')
        metadata = MetaData(bind=engine, schema=MunichData.schema)
        my_table = sqlalchemy.Table(table_name, metadata,
                                    autoload=True)
        if isinstance(primary_key, list):
            cons = PrimaryKeyConstraint(primary_key[0], primary_key[1], table=my_table)
        else:
            cons = PrimaryKeyConstraint(primary_key, table=my_table)
        cons.create()

        # define fields as not null
        print('_define fields as not null')
        if isinstance(primary_key, list):
            columns_to_upload.remove(primary_key[0])
            columns_to_upload.remove(primary_key[1])
        else:
            columns_to_upload.remove(primary_key)
        for colname in columns_to_upload:
            col = sqlalchemy.Column(colname, metadata)
            col.alter(nullable=False, table=my_table)


class DataForServerDictionaries:
    """
    this class helps to organise data for matching tables in server
    """
    COLUMNS = ['walcycdata_id', 'osm_id', 'start_point_id', 'end_point_id']
    PRIMARY_COLUMNS = ['walcycdata_id', 'osm_id']

    def __init__(self):
        # index store the relation to the refined network
        self.dict = {'index': [], 'walcycdata_id': [], 'osm_id': [], 'start_point_id': [], 'end_point_id': []}

    def find_relevant_data_to_update_dictionary(self, row, field):
        """
        The method get geo series of local object and matching OSM objects and separate  each local_object_osm_object
        :param row:
        :param field:car or cycle ('walcycdata_id_car' or 'walcycdata_id_cycle')
        :return:
        """
        # Matching one to one:
        if len(row['osm_ids']) == 1:
            self.__update_dictionaries(index=row['index'], walcycdata_id=row[field], osm_id=row['osm_ids'][0],
                                       start_point_id=row['start_point_id'], end_point_id=row['end_point_id'])
        # Matching one to many:
        else:
            # The first projection point belongs only to the first match
            self.__update_dictionaries(index=row['index'], walcycdata_id=row[field], osm_id=row['osm_ids'][0],
                                       start_point_id=row['start_point_id'], end_point_id=-1)
            for item in row['osm_ids'][1:-1]:
                self.__update_dictionaries(index=row['index'], walcycdata_id=row[field], osm_id=item,
                                           start_point_id=-1, end_point_id=-1)
            # The last projection point belongs only to the first match
            self.__update_dictionaries(index=row['index'], walcycdata_id=row[field], osm_id=row['osm_ids'][-1],
                                       start_point_id=-1, end_point_id=row['end_point_id'])

    def __update_dictionaries(self, index, walcycdata_id, osm_id, start_point_id, end_point_id):
        """
        It update the dictionary
        :param walcycdata_id:
        :param osm_id:
        :param start_point_id:
        :param end_point_id:
        :return:
        """
        self.dict['index'].append(index)
        self.dict['walcycdata_id'].append(walcycdata_id)
        self.dict['osm_id'].append(osm_id)
        self.dict['start_point_id'].append(start_point_id)
        self.dict['end_point_id'].append(end_point_id)
