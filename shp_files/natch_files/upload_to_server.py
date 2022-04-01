import geopandas as gpd
import glob
from sqlalchemy import create_engine, MetaData

if __name__ == '__main__':

    path_to_file = '*.shp'
    for file in glob.glob(path_to_file):
        name = file.split('.')[0]
        print(name)
        engine = create_engine('postgresql://research:1234@34.142.109.94:5432/walcycdata')
        gpd_data = gpd.read_file(file)
        if gpd_data.crs != "EPSG:4326":
            gpd_data.to_crs(crs="EPSG:4326", inplace=True)

        gpd_data.to_postgis(name=name, con=engine, schema='production',
                            if_exists='replace')
        print('{} is uploaded to server'.format(name))
