import pandas as pd
import numpy as np
import os
import time
import sys
import requests
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import json
from random import randint
from sqlalchemy import create_engine
from datetime import datetime, timedelta
sys.path.append('core')
from mysql import MYSQL


pd.options.mode.chained_assignment = None
db_connection = 'mysql+pymysql://douzi@traffic110:1qaz!QAZ2wsx@WSX@traffic110.mysql.database.azure.com/traffic'
# db_connection = 'mysql+pymysql://root:@localhost/accident'
conn = create_engine(db_connection)
df = pd.read_sql("""
                 SELECT a.ACCIDENT_NO, ACCIDENTDATE, ACCIDENTTIME, n.Lat, n.Long, a.SPEED_ZONE  FROM accident as a
                        LEFT JOIN node as n ON a.ACCIDENT_NO = n.ACCIDENT_NO
                        """, conn)

ms = MYSQL(host="traffic110.mysql.database.azure.com",user="douzi@traffic110",pwd="1qaz!QAZ2wsx@WSX",db="traffic")
# ms = MYSQL(host="localhost",user="root",pwd="",db="accident")

df['ACCIDENTDATE'] = pd.to_datetime(df['ACCIDENTDATE'])
df['ACCIDENTDATE_FOMAT'] = df['ACCIDENTDATE'].dt.strftime('%Y%m%d')
df['ACCIDENTTIME'] = pd.to_datetime(df['ACCIDENTTIME']).dt.strftime('%H:00:00')
df['Datetime'] = pd.to_datetime(df['ACCIDENTDATE'].apply(str) + ' ' + df['ACCIDENTTIME'])
df['Timestrap'] = df['Datetime'].values.astype(np.int64) // 10 ** 9

#https://api.weather.com/v1/geocode/-37.688/144.841/observations/historical.json?apiKey=6532d6454b8aa370768e63d6ba5a832e&startDate=20060101&endDate=20060102&units=e
base_url = 'https://api.weather.com/v1/geocode/'
format_url = '/observations/historical.json?units=e'
api_key = '&apiKey=6532d6454b8aa370768e63d6ba5a832e'

def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))


# df.reset_index(level=None, drop=True, inplace=True)
# for i in range(df.shape[0]):
df['snowing'] = 0
df['raining'] = 0
df['foggy'] = 0
df['smoke'] = 0
df['dust'] = 0
df['strong_winds'] = 0
df['wind_dir'] = ''
df['wind_speed'] = ''
df['temperature'] = 0

# for i in range(3):
for i in range(df.shape[0]):
    url = base_url + str(df['Lat'][i]) + '/' + str(df['Long'][i]) + format_url + '&startDate=' + \
          df['ACCIDENTDATE_FOMAT'][i] + '&endDate=' + df['ACCIDENTDATE_FOMAT'][i] + api_key
    print(url)
    print(df['ACCIDENT_NO'][i])
    print(df['Lat'][i], df['Long'][i])
    res = requests.get(url).json()

    while res['metadata']['status_code'] != 200:
        url = base_url + '-37.' + str(randint(0, 9999)) + '/' + '144.' + str(
            randint(0, 999)) + format_url + '&startDate=' + df['ACCIDENTDATE_FOMAT'][i] + '&endDate=' + \
              df['ACCIDENTDATE_FOMAT'][i] + api_key
        try:
            res = requests.get(url).json()
        except:
            pass


    w_date = []
    print(res['metadata']['status_code'])
    for j in range(len(res['observations'])):
        ts = int(res['observations'][j]['valid_time_gmt'])
        w_time = datetime.utcfromtimestamp(ts)
        w_time = w_time + timedelta(hours=11)
        w_date.append(w_time)
    getDate = nearest(w_date, df['Datetime'][i])
    for k in range(len(w_date)):
        if getDate == w_date[k]:
            df['wind_dir'][i] = res['observations'][k]['wdir_cardinal']
            df['wind_speed'][i] = res['observations'][k]['wspd']
            df['temperature'][i] = res['observations'][k]['temp']

    getWeather = ms.ExecQuery("""
                   SELECT atmosph_cond FROM ATMOSPHERIC_COND WHERE ACCIDENT_NO = '%s'
                   """ % (df['ACCIDENT_NO'][i]))
    get_weather = getWeather[0][0]
    if get_weather == 2:
        df['raining'][i] = 1
    elif get_weather == 3:
        df['snowing'][i] = 1
    elif get_weather == 4:
        df['foggy'][i] = 1
    elif get_weather == 5:
        df['smoke'][i] = 1
    elif get_weather == 6:
        df['dust'][i] = 1
    elif get_weather == 7:
        df['strong_winds'][i] = 1
    else:
        print('Sunny day')
    df.iloc[[i]].to_csv('my_csv.csv', mode='a', header=False)

df = df[['ACCIDENT_NO','wind_dir', 'wind_speed', 'temperature', 'snowing', 'raining', 'foggy', 'smoke', 'dust', 'strong_winds', 'ACCIDENTDATE', 'ACCIDENTTIME', 'Lat', 'Long', 'SPEED_ZONE']]
df.to_csv('accident_weather.csv')
# val = list(df.itertuples(index=False, name=None))
# val = ','.join(str(e) for e in val)


# try:
#     ms.ExecNonQuery("""
#                 INSERT INTO accident_weather
#                     (accident_no, wind_dir, wind_speed, temperature, snowing, raining, foggy, smoke, dust, strong_winds, accident_date, accident_time, lat, long, speed)
#                         VALUES %s
#                 """ % (val))
# except:
#     print('Add data error')
    # df.drop(index=i, inplace=True)