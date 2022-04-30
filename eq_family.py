# Read match_records.csv and generate lists of template families.
# The families will be plotted with time on X and location along strike (trench) on Y.
# The template event is a different symbol from the matches found by EqCorrScan, but the events are connected by lines.
'''
In order to plot templates according to distance along strike, use the 'cross track' calculation as given here:
https://www.movable-type.co.uk/scripts/latlong.html
'''
import os
import sys
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

IMAGE_DIR = '/proj/shumagin/gnelson/plots/detections'
MATCH_RECORD_FILE = '/proj/shumagin/gnelson/match_records.csv'
IMAGE_DIR = 'E:\Glenn Nelson\science\eq_gaps\plots\detections'
MATCH_RECORD_FILE = 'E:\Glenn Nelson\science\eq_gaps\match_records.csv'
#IMAGE_DIR = '.\detections'
STATION_FILE = 'E:\Glenn Nelson\science\eq_gaps\station_merge.csv'
EV_REGION_FILE = 'E:\Glenn Nelson\science\eq_gaps\ev_selected_region.csv'
DT_MATCH_VALUE = dt.timedelta(seconds=30)

from functools import wraps
from time import time
# use as a decorator to time functions
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
              (f.__name__, args, kw, te-ts))
        return result
    return wrap

def dt_match(dt1, dt2, dt_diff=DT_MATCH_VALUE):
    '''
    Compare datetime values. True if they differ by less than dt_diff
    :param dt1: datetime value
    :param dt2: datetime value
    :param dt_diff: datetime.timedelta value
    :return: True if match
    '''
    return abs(dt1 - dt2) < dt_diff

def haversine(ll1, ll2):
    '''
    Haversine method for great circle distance between 2 points on earth.
    See: https://www.movable-type.co.uk/scripts/latlong.html
    :param ll1: (longitude, latitude) numpy array in degrees
    :param ll2:  (longitude, latitude) numpy array in degrees
    :return: (a, c, d, bearing)
    a is the square of half the chord length
    c is the distance between points in radians
    d is the disance between points in km
    bearing is the initial bearing angle from ll1 to ll2
    '''
    # Using the haversine formula
    # phi is latitude in radians
    # theta is longitude in radians
    # convert to radians
    ll1rad = math.pi/180.0 * ll1
    ll2rad = math.pi/180.0 * ll2
    ldelta = ll2rad - ll1rad
    phi_delta = ldelta[1]   # latitude in radians
    theta_delta = ldelta[0] # longitude in radians (referneced paper uses lambda instead of theta)
    # c is the angular distance in radians; a is the square of half of the chord length
    a = math.sin(phi_delta/2)**2 + math.sin(theta_delta/2)**2 + math.cos(ll1rad[1]) * math.cos(ll2rad[1])
    c = math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c

    # initial bearing from ll1 to ll2
    y = math.sin(theta_delta) * math.cos(ll2rad[1])
    x = math.cos(ll1rad[1]) * math.sin(ll2rad[1]) - math.sin(ll1rad[1]) * math.cos(ll2rad[1]) * math.cos(theta_delta)
    theta = math.atan2(y, x)
    bearing = (theta * 180.0/math.pi +360) % 360

    return (a,c,d,theta)

def line_dist(ll1, ll2, ll3):
    '''
    Great circle defined by two long-lat points, ll1 and ll2.
    A point on the earth, ll3.
    Compute closest point on great circle from ll3.
    Compute distance along great circle and perpendicular distance from it.
    :param ll1: (longitude, latitude) one point on great circle
    :param ll2: (longitude, latitude) another point on great circle
    :param ll3: (longitude, latitude) point somewhere on earth
    :return: (dist along great circle, dist perpendicular) in km
    '''
    R = 6371E3     # radius of earth
    a12,c12,d12,bearing12 = haversine(ll1, ll2)
    a13,c13,d13,bearing13 = haversine(ll1, ll3)

    dang_xt = math.asin(math.sin(c13) * math.sin(bearing13-bearing12))
    d_xt = dang_xt * R  # cross-track dist (from LL3 to nearest point on LL1-LL2)
    dang_at = math.acos(math.cos(d13) / math.cos(dang_xt))
    d_at = dang_at * R  # along-track dist (to nearest point on LL1-LL2)
    return d_at,d_xt

class EqTemplates:
    # maintain a list of all templates and enable lookup of their event data
    def __init__(self, csv_file=EV_REGION_FILE):
        self.templ_csv = csv_file
        # open the file with pandas
        # it looks like this
        # time,longitude,latitude,depth,mag,region,template,templ_file
        # 2018-05-12T08:56:33.501000Z,-161.5485,54.3787,25500.0,2.2,0,2018-05-12T08:56:33.000000Z,2018_05_12t08_56_33.tgz
        df = pd.read_csv(self.templ_csv, dtype={'region':'string'})
        # convert 'time' to UTCDateTime
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
        new_df = self.df[self.df['templ_dt'].apply(lambda x: dt_match(ev_t, x, self.time_diff))]
        #print(new_df)
        return new_df

    def regionSelect(self, region='All'):
        '''
        Return a Dataframe of events in specified region.
        :param region: string: 'All' or other values, '0' to '7' for Shumagin study
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
    def __init__(self, path, templates, match_file=MATCH_RECORD_FILE):
        self.imageDir = path
        self.templates = templates
        self.match_file = match_file
        self.region_count = pd.DataFrame()
        self.index = -1 # row index in self.df
        self.region = 'All'
        self.savefile = False
        if not os.path.isfile(match_file):
            print('ERROR: {} must be created by gen_csv.py'.format(match_file))
            sys.exit(1)

        # self.df has regions assigned to each image and columns for 'quality' and 'is_new'
        self.df = self.readFile(match_file)
        self.df_select = self.df     # this will be the working view of self.df
        self.family_df = None
        # TODO: filtering already seen images should be optional so we can review old quality assignments
        # select images that do not yet have 'quality' assigned
        self.image_files = []
        print('MatchImages: {} images before removeSeen'.format(self.df.shape[0]))
        self.removeSeen(seen=True, unseen=False)    # only look at images we've already seen
        print('MatchImages: {} images'.format(len(self.image_files)))
        self.tallyRegions()

    @classmethod
    def parseImageFilename(cls, imgName):
        # extract template date and detection date
        f,x = imgName.split('.')
        t,d = f.split('_')
        templ_date = dt.datetime.strptime(t[len('Templ-'):], '%Y-%m-%dT%H-%M-%S')
        det_date = dt.datetime.strptime(d[len('Det-'):], '%Y-%m-%dT%H-%M-%S')
        return templ_date,det_date

    def readFile(self, file):
        date_cols = ['templ_dt', 'det_dt']
        df = pd.read_csv(file, dtype={'region':'string'}, parse_dates=date_cols)
        return df

    def getTemplateTimes(self):
        '''
        Generate list of unqiue template times
        :return:
        '''
        pass

    def getTemplateFamily(self, templ_dt):
        '''
        Get all rows associated with a template.
        The rows from match_records.csv also tell us if the event is template-self, another template, or new event.
        Also we get the quality and region values.
        :param templ_dt: template datetime
        :return: DataFrame with all the matching rows
        '''
        pass

    @timing
    def _findRegions(self, templates):
        # Match all image filenames with assigned region for the template
        # compare columns in 2 DF for matches: https://datascience.stackexchange.com/questions/33053/how-do-i-compare-columns-in-different-data-frames
        #print(self.df)
        cnt = 1
        # TODO: need vast improvement in speed
        # both templates and this use DF sorted by templ_dt
        # iterate over rows of self.df
        # lookup in templates.df, but don't use Pandas, instead just use a generator to find first match
        # when match is found, assign the region to current row.
        # no step forward in self.df, continue to assign region until row['templ_dt'] is different
        tdt = None
        for index,row in self.df.iterrows():
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
                tdt = templ_dt
                if not match.empty:
                    region = match.iloc[0]['region']
                    if not region in self.regionLookup:
                        self.regionLookup[region] = list()
                    self.regionLookup[region].append(row['filename'])
            except:
                print('ERROR in _findRegions: file={}'.format(row['filename']))

    def tallyRegions(self):
        '''
        Count of image files in each region. Fills a DataFrame with counts of all images in each region
        and unseen images in each.
        :return:
        '''
        # TODO: add a count of evaluated files
        # region_count is a Series, number of images in each region
        region_count = self.df['region'].value_counts()
        region_count.rename('total', inplace=True)
        print('Region\tCount\n{}'.format(region_count))
        # if review checkbox is off, df_select is only images not yet evaluated.
        # if review checkbox is on, df_select is all images.
        # TODO: next two lines should only be one. Figure out about chained indexing.
        unseen_df = self.df[self.df['quality'] == -1]
        unseen_count = unseen_df['region'].value_counts()
        unseen_count.rename('unseen', inplace=True)
        # TODO: add a row for 'All' regions
        print('Unseen\nRegion\tCount\n{}'.format(unseen_count))
        self.region_count = pd.concat([region_count, unseen_count], axis=1)

    def getRegionCounts(self):
        '''
        region_count index is the region name, columns are number of images: 'total' and 'unseen'
        :return: DataFrame populated by method tallyRegions
        '''
        if self.region_count.empty:
            self.tallyRegions()
        return self.region_count

    def getImageDir(self):
        return self.imageDir

    def removeSeen(self, seen=True, unseen=True):
        '''
        Remove some image files from list if they've been seen or not.
        Modifies self.image_files
        :param seen: if True, keep seen images, remove if False (inverse of expected behavior)
        :param unseen: if True, keep unseen images, remove if False (inverse of expected behavior)
        :return:
        '''
        if seen and unseen:     # keep all images
            self.df_select = self.df
        elif unseen:    # keep only unseen images
            self.df_select = self.df[self.df['quality'] == -1]
        else:           # keep only seen images
            self.df_select = self.df[self.df['quality'] != -1]
        print('removeSeen: seen={}, unseen={}, # images = {}'.format(seen, unseen, self.df_select.shape[0]))
        self.image_files = self.df_select['filename'].tolist()

    # TODO: probably not used
    def getNext(self, view_seen=False, view_unseen=True):
        '''
        If review==False, then get next image that has not yet been reviewed from self.df_select.
          If we've reached end of df_select, then return empty DF and caller will decide what to do.
        If review==True, then get next image from df_select. It may have been evaluated already or not.
          Caller will handle the UI setting of buttons.
          If we've reached end of df_select, then return empty DF and lett caller decide what to do.
        :param review:
        :return:
        '''
        self.index += 1
        retval = pd.DataFrame()     # empty DF
        nimage = self.df_select.shape[0]
        print('getNext: index={}, total={}'.format(self.index,nimage))
        if self.index >= nimage:
            print('getNext: no more images in current region, start at zero')
            self.index = 0
            # retval is empty DataFrame
        if view_seen:
            # get next image that has been reviewed or not
            retval = self.df_select.iloc[self.index]
        else:
            # look for next image that has not yet been reviewed
            while self.index < nimage:
                #print(self.df_select.iloc[self.index]['quality'])
                if self.df_select.iloc[self.index]['quality'] == -1:
                    retval = self.df_select.iloc[self.index]
                    break
                self.index += 1
        # if retval==df.empty, we've reached the end, caller of getNext must do something special
        return retval

    # TODO: probably not used
    def getPrev(self, view_seen=False, view_unseen=True):
        self.index -= 1
        # TODO: must test index for end of DF
        retval = pd.DataFrame()     # empty DF
        # skip to next unseen
        nimage = self.df_select.shape[0]
        print('getPrev: index={}, total={}'.format(self.index,nimage))
        if self.index < 0:
            print('getPrev: wrap around to last image')
            self.index = nimage-1
            # retval is empty DataFrame
        if view_seen:
            # get next image that has been reviewed or not
            retval = self.df_select.iloc[self.index]
        else:
            # look for next image that has not yet been reviewed
            while self.index >= 0:
                #print(self.df_select.iloc[self.index]['quality'])
                if self.df_select.iloc[self.index]['quality'] == -1:
                    retval = self.df_select.iloc[self.index]
                    break
                self.index -= 1
        # if retval==df.empty, we've reached the end, caller of getPrev must do something special
        return retval

    def filterRegion(self, region='All', seen=False, unseen=True, quality=None):
        '''
        Remove image files from list unless they match the region parameter.
        Region parameter is found in EqTemplates.
        :param region: 'All' or other values, '0' to '7' for Shumagin study
        :param unseen: True: only show images not yet reviewed; False: review all images
        :param quality: list of quality match values for review. If None, then review all images.
        :return:
        '''
        # TODO: should probably have a single method for filter that combines both region and 'seen'
        # modifies self.image_files
        self.removeSeen(seen=seen, unseen=unseen)
        self.index = -1
        if region == 'All':
            self.df_select = self.df
        else:
            self.df_select = self.df[self.df['region'] == region]
        # filter quality assessments when reviewing
        if seen and quality:
            # quality is a list of integers that we will review. 1 is no match, 4 is excellent match.
            if unseen:  # if also looking at unseen images, must add -1 to quality list
                quality.append(-1)
            self.df_select = self.df_select[self.df_select['quality'].isin(quality)]
        files = self.df_select['filename'].tolist()
        # self.image_files is list of all files that have not been reviewed for quality
        # files is list of all files in the region, regardless of review status
        # use set intersection to get only image_files in region that have not been reviewed
        files = set(self.image_files) & set(files)
        print('filterRegion: {} has {} images'.format(region, len(files)))
        self.image_files = sorted(list(files))
        return self.image_files

    def getFiles(self):
        return self.image_files

    def filter(self, quality=[3,4], is_new=[1,]):
        '''
        Filter the match events.
        :param quality: list of quality values to select (1 to 4)
        :param is_new: list. 1 is new events, 0 is events that are another template
        :return:
        '''
        self.df_select = self.df[(self.df['quality'].isin(quality)) & (self.df['is_new'].isin(is_new))]

    def family(self):
        '''
        Create DataFrame with template time and detection time.
        Column 1 is template_dt, Column 2 is a list of all associated det_dt
        :return:
        '''
        by_templ_dt = self.df_select.groupby(['templ_dt',])
        by_templ_list = []
        for tdt,frame in by_templ_dt:
            flist = frame['det_dt'].tolist()
            row = [tdt,flist]   # each row in new DF
            by_templ_list.append(row)
        # creates DF from dict. But the index is templ_dt and there is only one column that contains det_dt
        # creates DF from list. The index is integers, first column is templ_dt, second is det_dt
        self.family_df = pd.DataFrame.from_records(by_templ_list, columns=['templ_dt','det_dt'])
        return self.family_df

def getFamilyEvents(templates, matches):
    '''
    Each template has 0 or more matches.
    For each template, append the list of match times.
    The 'templates' object is modified with a new column in its DataFrame
    :param templates: instance of EqTemplates
    :param matches: instance of MatchImages
    :return: nothing
    '''
    # DF with two columns: templ_dt (single datetime) and det_dt (list of detection datetimes)
    detect_df = matches.family()
    # add detect_df column to templates DF
    # lookup template events based on time, add number of new events as new column
    # TODO: this is pretty inefficient
    templ_list = []
    for index,row in detect_df.iterrows():
        templ = templates.find(row['templ_dt'])
        if not templ.empty:
            templ_row = templ.iloc[0].tolist()
            templ_row.append(row['det_dt'])
            templ_list.append(templ_row)
        else:
            print('SKIP: no match for {}'.format(row['templ_dt']))

    df_new = pd.DataFrame(data=templ_list, columns=['time','longitude','latitude','depth','mag','region','template','templ_file','templ_dt','det_dt'])

def main():
    pass