import pandas as pd
import numpy as np
import os
import time
import sys
import json


class NegativeSampler:
    """
    Given features (positive examples), roads, times and the number of samples to draw, draws a sample
    """

    def __init__(self, num_samples, df, seed=None):

        self.num_samples = num_samples
        self.df = df

        if seed:
            np.random.seed(seed)

    def sample(self):
        segment_ids = self.df.index.to_series()

        altered = pd.DataFrame()
        num_to_sample = self.num_samples

        while num_to_sample > 0:
            samples = self.df[['timestamp', 'Route_No']].sample(n=num_to_sample, replace=True).reset_index()
            sample_timestamps, sample_segment_ids = self._mod(samples.copy(), segment_ids,
                                                              1.0 / (self.df.accident_counts + 1))
            # Create an index

            timestamp_strings = sample_timestamps.map(lambda x: str(int(x.timestamp()))).values
            segment_id_strings = sample_segment_ids.values.astype('str')

            fid = timestamp_strings + segment_id_strings
            alt = pd.DataFrame({
                'timestamp': sample_timestamps.values,
                'Route_No': sample_segment_ids.values,
                'segment_id': fid
            }).set_index('segment_id')

            altered = altered.append(alt)
            # Which happened before? They shouldn't get negative samples
            intersection = altered.index.intersection(self.df.index)
            idxer = intersection.get_values()

            # Drop samples where accidents occurred.
            altered = altered.drop(idxer)

            num_to_sample = self.num_samples - altered.shape[0]
        altered['target'] = 0

        ts = pd.DatetimeIndex(altered.timestamp)
        # altered = altered.reset_index()
        altered['DAY_OF_WEEK'] = ts.weekday + 1
        # altered['DAY_OF_WEEK'] = altered['DAY_OF_WEEK'].astype('int64') + 1
        altered['month'] = ts.month
        altered['hour'] = ts.hour

        # altered = altered.reset_index(drop=True)  # .drop(columns=['fid'])
        return altered

    def _mod(self, samples, segment_ids, w=None):
        # Get the current timestamps
        ts = pd.DatetimeIndex(samples.timestamp)

        # Hour, Day, Year
        hour = ts.hour.to_series()
        day = ts.dayofyear.to_series()
        year = ts.year.to_series()

        # Road ID
        segment_id = samples.Route_No.copy()

        sample_segment_ids = pd.Series()
        sample_timestamps = pd.Series()

        # Index of samples to mutate
        # feat_i = np.random.randint(0,3,size=samples.shape[0])
        feat_i = np.random.choice([0, 1, 2], size=samples.shape[0], p=[0.1, 0.3, 0.6])
        ##########################
        # i == 0
        # Change hour of day
        idx = feat_i == 0
        samp_i = samples.loc[idx]
        N = samp_i.shape[0]

        # Sample until we have all different hours of the day
        num_same = N
        new_hours = hour.loc[idx].values.copy()
        dif_idx = np.ones(N, dtype='bool')

        while num_same != 0:
            new_hours[dif_idx] = np.random.choice(24, size=num_same)
            dif_idx = new_hours == hour.loc[idx].values
            dif_idx = dif_idx | ((new_hours - 1) == hour.loc[idx].values)
            dif_idx = dif_idx | ((new_hours + 1) == hour.loc[idx].values)
            num_same = dif_idx.sum()

        # Create new timestamps
        new_timestamps = year[idx].apply(pd.Timestamp, args=(1, 1))
        new_timestamps = pd.DatetimeIndex(new_timestamps)
        new_timestamps += pd.TimedeltaIndex(day[idx] - 1, unit='D')  # same day
        new_timestamps += pd.TimedeltaIndex(new_hours, unit='H')  # new hour

        sample_segment_ids = sample_segment_ids.append(segment_id.loc[idx].copy())
        sample_timestamps = sample_timestamps.append(new_timestamps.to_series().copy())
        ##########################

        ##########################
        # i == 1
        # Change day of year
        idx = feat_i == 1
        samp_i = samples.loc[idx]
        N = samp_i.shape[0]

        is_leap_yr = ts[idx].is_leap_year

        # Sample until we have all different days of the year.
        num_same = N
        new_days = day.loc[idx].values.copy()
        dif_idx = np.ones(N, dtype='bool')
        while num_same != 0:
            # Pay attention to leap years
            dif_leap_yr = (dif_idx & is_leap_yr)
            dif_no_leap_yr = dif_idx & (~is_leap_yr)

            new_days[dif_leap_yr] = np.random.choice(np.arange(1, 367, dtype='int'), size=dif_leap_yr.sum())
            new_days[dif_no_leap_yr] = np.random.choice(np.arange(1, 366, dtype='int'), size=dif_no_leap_yr.sum())

            dif_idx = new_days == day.loc[idx].values
            num_same = dif_idx.sum()

        # Create new timestamps
        timestamps = year[idx].apply(pd.Timestamp, args=(1, 1))
        new_timestamps = pd.DatetimeIndex(timestamps)
        new_timestamps += pd.TimedeltaIndex(new_days - 1, unit='D')  # new day
        new_timestamps += pd.TimedeltaIndex(hour[idx], unit='H')  # same hour

        sample_segment_ids = sample_segment_ids.append(segment_id.loc[idx].copy())
        sample_timestamps = sample_timestamps.append(new_timestamps.to_series().copy())
        ##########################

        ##########################
        # i == 2
        # Change road
        idx = feat_i == 2
        samp_i = samples.loc[idx]
        N = samp_i.shape[0]

        num_same = N
        new_roads = segment_id.loc[idx].values.copy()
        dif_idx = np.ones(N, dtype='bool')

        while num_same != 0:
            new_roads[dif_idx] = segment_ids.sample(n=num_same, replace=True, weights=w).values
            dif_idx = new_roads == segment_id.loc[idx].values
            num_same = dif_idx.sum()

        sample_segment_ids = sample_segment_ids.append(pd.Series(new_roads))
        sample_timestamps = sample_timestamps.append(ts[idx].to_series().copy())

        sample_segment_ids = sample_segment_ids.astype('int64')
        return sample_timestamps, sample_segment_ids

# df = pd.read_csv('./positive_feature.csv')
# t = time.time()
# N = df.shape[0]*6
# ns = NegativeSampler(N,df)
# negative_examples = ns.sample()
# print(negative_examples)
