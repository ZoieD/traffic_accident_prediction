import pandas as pd
import sys

sys.path.append('../core')
from mysql import MYSQL

df_negative = pd.read_csv('./negative_0.csv')

df_negative['SPEED_ZONE'] = 0
df_negative['Light_Condition'] = 0
df_negative['ROAD_TYPE'] = ''
df_negative['DIRECTION_LOCATION'] = ''
df_negative['snowing'] = 0
df_negative['raining'] = 0
df_negative['foggy'] = 0
df_negative['smoke'] = 0
df_negative['dust'] = 0
df_negative['strong_winds'] = 0
df_negative['wind_dir'] = ''
df_negative['wind_speed'] = 0
df_negative['temperature'] = 0
df_negative['SURFACE_COND'] = 0
df_negative['NODE_TYPE'] = ''
df_negative['Deg_Urban_Name'] = ''
df_negative['target'] = 0
df_negative['accident_counts'] = 0
df_negative['date'] = df_negative['timestamp'].dt.strftime('%Y-%m-%d')

# ms = MYSQL(host="localhost",user="root",pwd="",db="accident")
ms = MYSQL(host="traffic110.mysql.database.azure.com",user="douzi@traffic110",pwd="1qaz!QAZ2wsx@WSX",db="traffic")

for i in range(df_negative.shape[0]):
# for i in range(2):
#     print(df_negative.iloc[i])
    getRoad = ms.ExecQuery("""
                   SELECT SPEED_ZONE, Light_Condition, ROAD_TYPE, DIRECTION_LOCATION, 
                   snowing, raining, foggy, smoke, dust, strong_winds, wind_dir, wind_speed,
                   temperature, SURFACE_COND, NODE_TYPE, Deg_Urban_Name
                   FROM positive_feature WHERE ACCIDENTDATE = '%s' AND Route_No = '%s'
                   LIMIT 1
                   """ % (df_negative['date'][i], df_negative['Route_No'][i]))
    if getRoad:
        df_negative['SPEED_ZONE'][i] = getRoad[0][0]
#         if
        df_negative['Light_Condition'][i] = getRoad[0][1]
        df_negative['ROAD_TYPE'][i] = getRoad[0][2]
        df_negative['DIRECTION_LOCATION'][i] = getRoad[0][3]
        df_negative['snowing'][i] = getRoad[0][4]
        df_negative['raining'][i] = getRoad[0][5]
        df_negative['foggy'][i] = getRoad[0][6]
        df_negative['smoke'][i] = getRoad[0][7]
        df_negative['dust'][i] = getRoad[0][8]
        df_negative['strong_winds'][i] = getRoad[0][9]
        df_negative['wind_dir'][i] = getRoad[0][10]
        df_negative['wind_speed'][i] = getRoad[0][11]
        df_negative['temperature'][i] = getRoad[0][12]
        df_negative['SURFACE_COND'][i] = getRoad[0][13]
        df_negative['NODE_TYPE'][i] = getRoad[0][14]
        df_negative['Deg_Urban_Name'][i] = getRoad[0][15]
    else:
        getRoad1 = ms.ExecQuery("""
                   SELECT snowing, raining, foggy, smoke, dust, strong_winds, wind_dir, 
                   wind_speed, temperature, SURFACE_COND
                   FROM positive_feature WHERE ACCIDENTDATE = '%s'
                   LIMIT 1
                   """ % (df_negative['date'][i]))
        if getRoad1:
            df_negative['snowing'][i] = getRoad1[0][0]
            df_negative['raining'][i] = getRoad1[0][1]
            df_negative['foggy'][i] = getRoad1[0][2]
            df_negative['smoke'][i] = getRoad1[0][3]
            df_negative['dust'][i] = getRoad1[0][4]
            df_negative['strong_winds'][i] = getRoad1[0][5]
            df_negative['wind_dir'][i] = getRoad1[0][6]
            df_negative['wind_speed'][i] = getRoad1[0][7]
            df_negative['temperature'][i] = getRoad1[0][8]
            df_negative['SURFACE_COND'][i] = getRoad1[0][9]
            getRoad2 = ms.ExecQuery("""
                       SELECT SPEED_ZONE, Light_Condition, ROAD_TYPE, DIRECTION_LOCATION,
                       NODE_TYPE, Deg_Urban_Name
                       FROM positive_feature WHERE Route_No = '%s'
                       LIMIT 1
                       """ % (df_negative['Route_No'][i]))
            if getRoad2:
                df_negative['SPEED_ZONE'][i] = getRoad2[0][0]
                df_negative['Light_Condition'][i] = getRoad2[0][1]
                df_negative['ROAD_TYPE'][i] = getRoad2[0][2]
                df_negative['DIRECTION_LOCATION'][i] = getRoad2[0][3]
                df_negative['NODE_TYPE'][i] = getRoad2[0][4]
                df_negative['Deg_Urban_Name'][i] = getRoad2[0][5]
            else:
                df_negative.drop(index=[i], inplace=True)
        else:
            df_negative.drop(index=[i], inplace=True)
    df.iloc[[i]].to_csv('negative_feature.csv', mode='a', index=False)
# df_negative.to_csv('./negative_feature.csv', index=False)