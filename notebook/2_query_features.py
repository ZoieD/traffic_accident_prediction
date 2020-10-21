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
print("start")
df = pd.read_sql("""
        SELECT a.ACCIDENT_NO, a.ACCIDENTTIME, a.DAY_OF_WEEK, a.SPEED_ZONE, a.ACCIDENTDATE, 
     a.Light_Condition, l.ROAD_TYPE, l.DIRECTION_LOCATION,l.DISTANCE_LOCATION,
     w.snowing, w.raining, w.foggy, w.smoke, w.dust, w.strong_winds, w.wind_dir, 
     w.wind_speed, w.temperature, s.SURFACE_COND, n.NODE_TYPE, n.Deg_Urban_Name, c.Route_No
        FROM accident a
            LEFT JOIN accident_location as l ON l.ACCIDENT_NO = a.ACCIDENT_NO
            LEFT JOIN accident_weather as w ON w.ACCIDENT_NO = a.ACCIDENT_NO
            LEFT JOIN node n ON n.ACCIDENT_NO = a.ACCIDENT_NO
            LEFT JOIN road_surface_cond s ON s.ACCIDENT_NO = a.ACCIDENT_NO
            LEFT JOIN accident_chainage c ON c.Node_Id = a.NODE_ID
            WHERE a.NODE_ID>0
                        """, conn)
print("End")
df.to_csv('static_feature.csv', index=False)