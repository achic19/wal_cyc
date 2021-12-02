from sqlalchemy import create_engine, MetaData
import sqlalchemy
from migrate.changeset.constraint import PrimaryKeyConstraint
from geopandas import GeoDataFrame


class MunichData:
    engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')
    schema = 'production'
    crs = "EPSG:3857"
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

