import numpy as np
import geopandas
import pathlib
import pandas as pd
import os
import matplotlib.pyplot as plt

def get_value_maps():
    WeekDay={'1':'sunday','2':'monday','3':'tuesday','4':'wednesday','5':'thursday','6':'friday','7':'saturday'}
    RoadCondition = {'0':'dry', '1':'wet','2':'winter_slippery'}
    LightCondition = {'0':'daylight','1':'twilight','2':'darkness'}
    BicyclesInvolved={'0':'no','1':'yes'}
    InjuryType = {'1':'death','2':'serious','3':'light'}
    return [WeekDay,RoadCondition,LightCondition,InjuryType,BicyclesInvolved]

def preprocess_incidents(datapath,savepath):
    if os.path.isfile(savepath):
        df = pd.read_pickle(savepath)
    else:
        path_object = pathlib.Path(datapath)
        df = geopandas.read_file(path_object)
        df = df.rename(columns={"IstRad": "BicyclesInvolved", "UWOCHENTAG": "WeekDay", "ULICHTVERH":"LightCondition","STRZUSTAND":"RoadCondition","UKATEGORIE":"InjuryType"})
        df = df.loc[df['BicyclesInvolved'] == '1'] #only incidents involving bicycles
        df = df[['WeekDay','RoadCondition','LightCondition','InjuryType','XGCSWGS84','YGCSWGS84','geometry']]
        df.to_pickle("germany_bicycle_incidents_2020.pkl")
    return df


if __name__=='__main__':
    # https://unfallatlas.statistikportal.de/_opendata2021.html
    incidentsfile = r'D:\Work\WalCycData\Data\AccidentData\Shapefile\Unfallorte2020_LinRef.shp'
    savepath = 'germany_bicycle_incidents_2020.pkl'
    df = preprocess_incidents(incidentsfile, savepath)   
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df.XGCSWGS84, df.YGCSWGS84))
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    ax = world[world.name == 'Germany'].plot(
        color='white', edgecolor='black')

    gdf.plot(ax=ax, color='red',markersize=1)
    plt.show()
