import datetime as dt
import pandas as pd
from config import EV_REGION_FILE
from my_util import dt_match

class EqTemplates:
    # maintain a list of all templates and enable lookup of their event data
    def __init__(self, csv_file=EV_REGION_FILE):
        self.templ_csv = csv_file
        # open the file with pandas
        # it looks like this
        # time,longitude,latitude,depth,mag,region,template,templ_file
        # 2018-05-12T08:56:33.501000Z,-161.5485,54.3787,25500.0,2.2,0,2018-05-12T08:56:33.000000Z,2018_05_12t08_56_33.tgz
        df = pd.read_csv(self.templ_csv, dtype={'region':'string'})
        # convert 'time' to datetime - tstr[:19] makes sure to ignore the fractional seconds value
        df['templ_dt'] = df['time'].apply(lambda tstr: dt.datetime.strptime(tstr[:19], '%Y-%m-%dT%H:%M:%S'))
        df['depth'] = df['depth'].div(1000.0)
        self.df = df.sort_values(by=['templ_dt'])
        print(self.df)

    def getDF(self):
        return self.df

    def find(self, ev_t, time_diff_sec=30, dt_format='%Y-%m-%dT%H-%M-%S'):
        '''
        Match ev_t with template_time. Return a Dataframe of matches.
        Use this to lookup a template event and retrieve location.
        Or use it to discover if a detection is another event that is already present as a template.
        We're usually processing PNG images of match_filter detections. The filenames contain
        both template datetime and detection datetime:
        PNG filename example: Templ-2018-05-19T08-14-07_Det-2018-05-22T05-24-26.png
        :param ev_t: datetime object, usually an event datetime from a match_filter PNG filename.
        :param time_diff_sec: match if less than this value.
        :param dt_format: not used?
        :return: a Dataframe. Expect only one row in the df.
        '''
        # return the data from pandas with lat-long and region number
        self.time_diff = dt.timedelta(seconds=time_diff_sec) # seconds
        new_df = self.df[self.df['templ_dt'].apply(lambda t_time: dt_match(ev_t, t_time, self.time_diff))]
        #print(new_df)
        return new_df

    def regionSelect(self, region='All'):
        '''
        Return a Dataframe of events in specified region.
        Regions are used to break a study area into smaller parcels in order to limit the number
        of templates that are used in Tribe.detect method. Regions are arbitrary and they can be named
        with any string. I chose values from '0' to '7' for the Shumagin study.
        :param region: string: 'All' or other values, e.g., '0' to '7'
        :return: Dataframe that is a subset of all events
        '''
        if region == 'All':
            return self.df
        else:
            new_df = self.df[self.df['region'] == region]
            return new_df