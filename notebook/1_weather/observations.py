import json
from datetime import datetime

import requests

API_ENDPOINT='https://dosszbxtc8.execute-api.ap-southeast-2.amazonaws.com/prod'


def get_observation_types():
    """
    Gets all possible observation types
    """

    response = requests.get('{}/list-obs-types'.format(API_ENDPOINT))
    return json.loads(response.content)


def get_stations():
    """
    Gets all stations
    """

    response = requests.get('{}/list-stations'.format(API_ENDPOINT))
    return json.loads(response.content)


def get_observation(start, end, interval, station, datatypes, destination):
    """
    :type start: datetime
    :param start: The start of my arguments.
    :type end: datetime
    :param end: The second of my arguments.
    :type interval: int [10, 15, 30, 60]
    :param interval: The second of my arguments.
    :type end: datetime
    :param end: The second of my arguments.

    :returns: (bool) indicating whether download of csv to destination was successful
    """

    def format_date(date):
        return date.strftime("%Y-%m-%d %H:%M:%S")

    datatypes = ','.join(datatypes)

    response = requests.get(
        '{}/raw-obs?station={}&start={}&end={}&interval={}&datatypes={}'.format(
            API_ENDPOINT, station, format_date(start), format_date(end), interval, datatypes
        ), stream=True
    )

    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True

    return False