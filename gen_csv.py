# Generate CSV files required by eq_match_tool.
"""
CSV fields: file, template_dt, detection_dt, region, quality, is_new
file: the detection image filename, example: Templ-2018-05-22T06-52-03_Det-2018-11-27T10-12-19.png
templ_dt: template datetime, no fractional seconds, example: 2018-05-22 06:52:03
det_dt: detection datetime, no fractional seconds, example: 2018-11-27 10:12:19
region: we use number values from 0 to 7, but they are interpreted as strings
quality: we use number values from 1 to 4
is_new: 1 means True, 0 means this detection is an existing template (perhaps itself)

region: In the Shumagin study we select events within a rectangular region that includes
the trench and downdip. This region is further broken down into smaller rectangles along strike
in order to run EqCorrScan with smaller numbers of templates for detection.

quality: 4 is a very good match, 3 is a "maybe" match, 2 is a "maybe not" match, and 1 is "no match".
The file has a value of -1 if the detection has not been evaluated.
"""
import csv
import os
import argparse
import traceback
import pandas as pd
import datetime as dt
from my_util import dt_match

# INPUT FILES
IMAGE_DIR = '/proj/shumagin/gnelson/plots/detections'
IMAGE_DIR = 'E:\Glenn Nelson\science\eq_gaps\plots\detections'
#IMAGE_DIR = '.\detections'
STATION_FILE = 'E:\Glenn Nelson\science\eq_gaps\station_merge.csv'
EV_REGION_FILE = 'E:\Glenn Nelson\science\eq_gaps\ev_selected_region.csv'
# OUTPUT FILE
MATCH_RECORD_FILE = '/proj/shumagin/gnelson/match_records.csv'
MATCH_RECORD_FILE = 'E:\Glenn Nelson\science\eq_gaps\match_records.csv'

from functools import wraps
from time import time
# use as a decorator to time functions
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        #print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        print('func:%r: took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

class EqTemplates:
    # maintain a list of all templates and enable lookup of their event data
    #time_diff = dt.timedelta(seconds=30)

    def __init__(self, csv_file=EV_REGION_FILE):
        self.templ_csv = csv_file
        # open the file with pandas
        # it looks like this
        # time,longitude,latitude,depth,mag,region,template,templ_file
        # 2018-05-12T08:56:33.501000Z,-161.5485,54.3787,25500.0,2.2,0,2018-05-12T08:56:33.000000Z,2018_05_12t08_56_33.tgz
        df = pd.read_csv(self.templ_csv, dtype={'region':'string'})
        # convert 'time' to datetime, make sure to eliminate fractional seconds, e.g., tstr[:19]
        df['templ_dt'] = df['time'].apply(lambda tstr: dt.datetime.strptime(tstr[:19], '%Y-%m-%dT%H:%M:%S'))
        df['depth'] = df['depth'].div(1000.0)
        self.df = df.sort_values(by=['templ_dt'])
        print(self.df)

    def getDF(self):
        return self.df

    def find(self, ev_t, time_diff_sec=30, dt_format='%Y-%m-%dT%H-%M-%S'):
        '''
        Match ev_t with template_time. Return a DataFrame of matches.
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
        new_df = self.df[self.df['templ_dt'].apply(lambda x: dt_match(ev_t, x, self.time_diff))]
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

class MatchImages:
    '''
    Images have filenames like this: Templ-2018-11-17T03-53-30_Det-2018-05-08T14-24-02.png
    Contains df (Pandas DataFrame) with templ_dt and region as columns.
    Method _findRegion will lookup the region by searching EqTemplates object.
    '''
    def __init__(self, path, templates, match_file, exclude_self=True):
        '''
        :param path: directory of all the detection image files
        :param templates: EqTemplates object that contains info about all the templates we have used
        :param match_file: filename for output of the detection image CSV 'database'
        :param exclude: do not write template self-detection images into match_file
        '''
        self.imageDir = path
        self.templates = templates
        self.match_file = match_file
        if os.path.isfile(match_file):
            self.df_old = self.readFile(match_file)
            self.matchFiles = self.df_old['filename'].tolist()
            print('MatchImages: {} is old'.format(match_file))
        else:
            self.df_old = pd.DataFrame()    # starting fresh, nothing to do here
            self.matchFiles = None
            print('MatchImages: {} is new'.format(match_file))

        # TODO: keep full list always, keep separate filtered list
        # TODO: regionLookup is dumb, should just merge with allFiles
        self.regionLookup = {}
        self.allFiles = None
        self.notSeenFiles = None
        self.files = None   # this is the current working set after all filters
        newfiles = []
        with os.scandir(self.imageDir) as ls:
            for item in ls:
                if item.is_file():
                    filename = str(item.name)
                    if self.matchFiles and filename in self.matchFiles:
                        pass    # already have filename in existing list
                    else:
                        newfiles.append(filename)
        print('MatchImages: new files = {}'.format(len(newfiles)))
        if newfiles:
            self.newfiles = sorted(newfiles)   # it will sort on template date
            rows = []
            for file in self.newfiles:
                templ_dt, det_dt = self.parseImageFilename(file)
                f_dict = {'filename':file, 'templ_dt':templ_dt, 'det_dt':det_dt, 'region':'-',
                          'quality':-1, 'is_new':-1}
                rows.append(f_dict)
            print('findRegions CALL')
            rows = self._findRegions(rows, templates)
            print('findRegions DONE')
            df_new = pd.DataFrame(data=rows, columns=['filename','templ_dt','det_dt','region','quality','is_new'])
        else:
            # TODO: using empty DF as a marker seems ugly
            df_new = pd.DataFrame()
        # TODO: messy tests here. df_old may be empty, df_new may be empty, or both may be not empty
        if self.df_old.empty:
            self.savefile = True
            self.df = df_new
        elif df_new.empty:
            self.savefile = False
            self.df = self.df_old
        else:
            self.savefile = True
            self.df = pd.concat([self.df_old,df_new])
        self.df = self.df.sort_values(by=['templ_dt'])
        print(self.df.iloc[0])
        #print(self.df.iloc[1])
        #print(self.df.iloc[2])
        print(self.df.iloc[-1])
        self.allFiles = sorted(self.df['filename'].tolist())
        self.files = self.allFiles

    @classmethod
    def parseImageFilename(cls, imgName):
        try:
            # extract template date and detection date
            f,x = imgName.split('.')
            t,d = f.split('_')
            templ_date = dt.datetime.strptime(t[len('Templ-'):], '%Y-%m-%dT%H-%M-%S')
            det_date = dt.datetime.strptime(d[len('Det-'):], '%Y-%m-%dT%H-%M-%S')
        except:
            print('FATAL: parseImageFilename: {}'.format(f))
            raise Exception('abort program')
        return templ_date,det_date

    @timing
    def _findRegions(self, rows, templates):
        '''
        Match all image filenames with assigned region for the template
        :param rows: list of dict, each dict is a row of image info
        :param templates: EqTemplate object, contains DF of all templates and their parameters
        :return:
        '''
        # compare columns in 2 DF for matches: https://datascience.stackexchange.com/questions/33053/how-do-i-compare-columns-in-different-data-frames
        cnt = 1
        # TODO: need vast improvement in speed
        # both templates and this use DF sorted by templ_dt
        # iterate over rows of self.df
        # lookup in templates.df, but don't use Pandas, instead just use a generator to find first match
        # when match is found, assign the region to current row.
        # no step forward in self.df, continue to assign region until row['templ_dt'] is different
        # update a row while using iterrows: itterows returns a copy of the row (no surprise),
        # but you are allowed to update the df: https://stackoverflow.com/questions/25478528/updating-value-in-iterrow-for-pandas
        # And the stackoverflow article recommends 'at' instead of 'loc'.
        tdt = None
        region = '-'
        print('findRegions has {} rows'.format(len(rows)))
        for row in rows:
            if cnt % 250 == 0:
                print('findRegions: {}'.format(cnt))
            cnt += 1
            try:
                templ_dt = row['templ_dt']
                if tdt and templ_dt == tdt:
                    # we already have the template that matches this event
                    # don't waste time with templates.find()
                    #print('cnt={}, dt={}'.format(cnt,templ_dt))
                    pass
                else:
                    match = templates.find(templ_dt)  # return a DF of record from EV_SELECTED_REGION
                    print('new match at {}, dt={}'.format(cnt,match.iloc[0]['templ_dt']))
                    #print('new match at {}, dt={}'.format(cnt,templ_dt))
                tdt = templ_dt
                if not match.empty:
                    region = match.iloc[0]['region']
                    #self.df.at[index, 'region'] = region    # store the region with the detection record
                    row['region'] = region  # store the region with the detection record
                    if not region in self.regionLookup:
                        self.regionLookup[region] = list()
                    self.regionLookup[region].append(row['filename'])
            except:
                print('ERROR in _findRegions: file={}'.format(row['filename']))
                print(traceback.format_exc())

        return rows

    def getImageDir(self):
        return self.imageDir

    def selectNotSeen(self):
        '''
        Return DF of files that have no match quality value
        :return:
        '''
        return self.df[self.df['quality'] == -1]

    def exclude(self, file):
        '''
        Exclude some templates from processing.
        The purpose of this is to eliminate events from EqTransformer until we sort out what's wrong.
        :param file: CSV file of events
        :return:
        '''
        events = []
        with open(file, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ev_dt = dt.datetime.strptime(row['time'][:19], '%Y-%m-%dT%H:%M:%S')
                events.append(ev_dt)
        print('exclude events: {}'.format(len(events)))
        # TODO: this might be too slow
        bad_df = self.df[self.df['templ_dt'].isin(events)]
        print('exclude detections count: {}'.format(bad_df.shape[0]))
        good_df = self.df[~self.df['templ_dt'].isin(events)]
        print('good df count: {}'.format(good_df.shape[0]))
        self.df = good_df
        self.savefile = True
        
    def exclude_self(self, time_diff_sec=30):
        '''
        Remove self-detection images: a template matches itself
        '''
        time_diff = dt.timedelta(seconds=time_diff_sec) # seconds
        print('exclude_self:')
        print('self.df has {} rows'.format(self.df.shape[0]))
        self.df = self.df[~self.df.apply(lambda x: dt_match(x['templ_dt'], x['det_dt'], time_diff), axis=1)]
        print('self.df has {} rows'.format(self.df.shape[0]))

    def readFile(self, file=MATCH_RECORD_FILE):
        date_cols = ['templ_dt', 'det_dt']
        df = pd.read_csv(file, dtype={'region':'string'}, parse_dates=date_cols)
        return df

    def save(self):
        if self.savefile:
            self.df.to_csv(self.match_file, index=False)
        else:
            print('MatchImages.save: no update to file')

    def getFiles(self):
        return self.files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build list of detection image files for evaluation by eq_match_tool')
    #parser.add_argument('-h','--help')
    parser.add_argument('-x', dest='bad_file', type=str, default=None, help='CSV event file to exclude')
    args = parser.parse_args()

    templates = EqTemplates()
    matchImages = MatchImages(IMAGE_DIR, templates, MATCH_RECORD_FILE)
    if args.bad_file:
        matchImages.exclude(args.bad_file)
    # remove any template self-detection
    matchImages.exclude_self()
    matchImages.save()


