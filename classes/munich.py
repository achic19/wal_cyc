from sqlalchemy import create_engine, MetaData
import sqlalchemy
from migrate.changeset.constraint import PrimaryKeyConstraint
from geopandas import GeoDataFrame
import pandas as pd


class MunichData:
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')
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
    def table_for_server(based_data: GeoDataFrame) -> tuple[GeoDataFrame, GeoDataFrame]:
        """
        Create two new table that keep only the cycle_id and car_id with the matching osm
        :param based_data:
        :return:
        """
        print('table_for_server')
        print('_preprocessing tasks')
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
        # For the new table, store only rows with cycles or cars data counts
        based_data = based_data[(based_data['cyclecount_count'] != 0) | (based_data['carcount_count'] != 0)]
        # create new dictionary
        print('create new dictionaries of DataForServerDictionaries class')
        data_dictionary_car = DataForServerDictionaries()
        data_dictionary_cycle = DataForServerDictionaries()

        # for each row check count for each type and if is it not zero update the relevant dictionary
        print('_update data_dictionary_cycle')
        based_data.apply(
            lambda x: data_dictionary_cycle.find_relevant_data_to_update_dictionary(x, 'walcycdata_id_cycle'), axis=1)
        print('_update data_dictionary_car')
        based_data.apply(
            lambda x: data_dictionary_car.find_relevant_data_to_update_dictionary(x, 'walcycdata_id_car'), axis=1)

        # Make the new dictionaries into a databases
        print('_make the new dictionaries into a databases')
        return GeoDataFrame(data_dictionary_cycle.dict), GeoDataFrame(data_dictionary_car.dict)

    @staticmethod
    def data_to_server(data_to_upload: GeoDataFrame,
                       columns_to_upload: list,
                       table_name: str
                       ):
        """
        Upload GeoDataFrame  into schema in engine
        and only with the columns specified in the columns_to_upload parameter
        :param table_name:
        :param data_to_upload:
        :param columns_to_upload:
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
        data_to_upload.to_postgis(name=table_name, con=MunichData.engine, schema=MunichData.schema,
                                  if_exists='replace',
                                  dtype={'walcycdata_last_modified': sqlalchemy.types.DateTime})
        # define primary key
        print('_define primary key')
        metadata = MetaData(bind=MunichData.engine, schema=MunichData.schema)
        my_table = sqlalchemy.Table(table_name, metadata, autoload=True)
        cons = PrimaryKeyConstraint('walcycdata_id', table=my_table)
        cons.create()

        # define fields as not null
        print('_define fields as not null')
        columns_to_upload.remove('walcycdata_id')
        for col in columns_to_upload:
            col = sqlalchemy.Column(col, metadata)
            col.alter(nullable=False, table=my_table)


class DataForServerDictionaries:
    """
    this class helps to organise data for matching tables in server
    """

    def __init__(self):
        self.dict = {'walcycdata_id': [], 'osm_id': [], 'start_point_id': [], 'end_point_id': []}

    def find_relevant_data_to_update_dictionary(self, row, field):
        """
        The method get geo series of local object and matching OSM objects and separate  each local_object_osm_object
        :param row:
        :param field:car or cycle ('walcycdata_id_car' or 'walcycdata_id_cycle')
        :return:
        """
        # The first condition ensures that the current local object has is related to cars of cycles database
        if row[field] != -1:
            # Matching one to one:
            if len(row['osm_ids']) == 1:
                self.__update_dictionaries(walcycdata_id=row[field], osm_id=row['osm_ids'][0],
                                           start_point_id=row['start_point_id'], end_point_id=row['end_point_id'])
            # Matching one to many:
            else:
                # The first projection point belongs only to the first match
                self.__update_dictionaries(walcycdata_id=row[field], osm_id=row['osm_ids'][0],
                                           start_point_id=row['start_point_id'], end_point_id=-1)
                for item in row['osm_ids'][1:-1]:
                    self.__update_dictionaries(walcycdata_id=row[field], osm_id=item,
                                               start_point_id=-1, end_point_id=-1)
                # The last projection point belongs only to the first match
                self.__update_dictionaries(walcycdata_id=row[field], osm_id=row['osm_ids'][-1],
                                           start_point_id=-1, end_point_id=row['end_point_id'])

    def __update_dictionaries(self, walcycdata_id, osm_id, start_point_id, end_point_id):
        """
        It update the dictionary
        :param walcycdata_id:
        :param osm_id:
        :param start_point_id:
        :param end_point_id:
        :return:
        """
        self.dict['walcycdata_id'].append(walcycdata_id)
        self.dict['osm_id'].append(osm_id)
        self.dict['start_point_id'].append(start_point_id)
        self.dict['end_point_id'].append(end_point_id)
