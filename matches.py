import os
import sys
import csv
import datetime as dt
import pandas as pd
from config import MATCH_RECORD_FILE
from my_util import dt_match

class MatchImages:
    '''
    Images have filenames like this: Templ-2018-11-17T03-53-30_Det-2018-05-08T14-24-02.png.
    The template datetime can be used to lookup template event info in an earthquake event file,
    such as QML or CSV file. The detection datetime can be used to determine if this is an existing event
    (it is found in the event file) or if it is a new event. Also detection datetimes can be used
    to construct a Family - events that are matches to a template.
    This class contains df (Pandas DataFrame) with templ_dt and region as columns.
    Method _findRegion will lookup the region by searching EqTemplates object.
    '''
    # TODO: should be able to instantiate with only match_file argument.
    def __init__(self, path, templates, match_file=MATCH_RECORD_FILE):
        self.imageDir = path
        self.templates = templates  # object of class EqTemplates
        self.match_file = match_file
        self.region_count = pd.DataFrame()
        self.index = -1 # row index in self.df
        self.region = 'All'
        self.savefile = False
        if not os.path.isfile(match_file):
            print('ERROR: {} must be created by gen_csv.py'.format(match_file))
            # TODO: should not call sys.exit, should throw exception
            sys.exit(1)

        # self.df has regions assigned to each image and columns for 'quality' and 'is_new'
        # quality ranges from 1 to 4 (best) or -1 if detection image has not been evaluated
        # is_new =0 if the detected event is another template, =1 if it is new, or -1 if not yet evaluated
        self.df = self.readFile(match_file)
        self.df_select = self.df     # this will be the working view of self.df
        self.family_df = None
        # TODO: filtering already seen images should be optional so we can review old quality assignments
        # select images that do not yet have 'quality' assigned
        self.image_files = []
        print('MatchImages: {} images before removeSeen'.format(self.df.shape[0]))
        # select images that do not yet have 'quality' assigned
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

    def getSelectedDF(self):
        return self.df_select

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

    # TODO: should rename as keepImages or useImages
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
        # TODO: not using parameter view_unseen, but you should
        If view_seen==True, then get next image whether already seen or not.
        If view_seen==False, then only get next unseen image.
        If we've reached end of df_select, then return empty DF and let caller decide what to do.
        :param view_seen: return next image that has already been seen
        :param view_unseen: return next image that has not yet been seen
        :return: DataFrame where first row contains the image data or empty DF if none.
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
        '''
        See comments for getNext method.
        :param view_seen:
        :param view_unseen:
        :return:
        '''
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
        Creates self.df_select from full DataFrame self.df.
        :param region: 'All' or other values, '0' to '7' for Shumagin study
        :param unseen: True: only show images not yet reviewed; False: review all images
        :param quality: list of quality match values for review. If None, then review all images.
        :return: list of the image filenames
        '''
        # TODO: should probably have a single method for filter that combines both region and 'seen'
        # modifies self.image_files
        self.removeSeen(seen=seen, unseen=unseen)
        self.index = -1
        if region == 'All':
            self.df_select = self.df
        else:
            if isinstance(region, str):
                region = [region,]
            self.df_select = self.df[self.df['region'].isin(region)]
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

    def saveMatch(self, match_file, templ_date, det_date, region, quality, is_new):
        '''
        Stores our evaluation of the match quality in DataFrame.
        Also adds the match_file name to internal list so that we know we've seen it during this session.
        :param match_file: filename of match image file
        :param templ_date: date of the template (also embedded in the filename)
        :param det_date: date of the detection (also embedded in the filename)
        :param region: designation for the region that the template is located in
        :param quality: evaluated quality of detection, a number from 1 to 4
        :param is_new: False: if detection is another template that we already have.
        :return:
        '''
        #print('saveMatch: {},{},{},{},{},{}'.format(match_file,templ_date,det_date,region,quality,is_new))
        self.savefile = True
        tstr = templ_date.strftime('%Y-%m-%d %H:%M:%S')
        dstr = det_date.strftime('%Y-%m-%d %H:%M:%S')
        match = {'file':match_file,'templ_date':tstr,'detect_date':dstr,'quality':quality,'is_new':is_new,'region':region}
        # TODO: why am I updating both df_select and df?
        self.df_select.iloc[self.index, self.df_select.columns.get_loc('is_new')] = is_new
        self.df_select.iloc[self.index, self.df_select.columns.get_loc('quality')] = quality
        self.df.iloc[self.df['filename']==match_file, self.df_select.columns.get_loc('is_new')] = is_new
        self.df.iloc[self.df['filename']==match_file, self.df_select.columns.get_loc('quality')] = quality
        print('saveMatch: {}'.format(self.df_select.iloc[self.index]))

    def filter(self, quality=(3,4), is_new=(1,)):
        '''
        Filter the match events.
        If events have not been 'seen' (not yet evaluated), then they will be excluded since the event
        quality value will be -1.
        :param quality: list of quality values to select (1 to 4)
        :param is_new: list. 1 is new events, 0 is events that are another template
        :return:
        '''
        self.df_select = self.df[(self.df['quality'].isin(quality)) & (self.df['is_new'].isin(is_new))]

    def remove_selected(self):
        '''
        Remove matches in df_select from df. Not expected to be used, but needed to get rid of matches
        that had to be redone.
        :return:
        '''
        # Union, intersection, and difference with DataFrames:
        # https://www.kdnuggets.com/2019/11/set-operations-applied-pandas-dataframes.html
        # Use difference
        new_df = self.df[self.df.filename.isin(self.df_select.filename) == False]
        return new_df

    def quality_stats(self):
        '''
        Generate statistics of the quality of each template.
        Tally the assigned quality values for each match.
        Each template timestamp is a key in a dict. The dict values are a list of 5 numbers.
        Index 0 is the mean value of the quality for all matches for the template.
        Index 1 to 4 is the count of number of matches for each quality level 1 to 4.
        :return: dict for each template timestamp
        '''
        templ_stats = dict()
        # next line generates a count of each quality rating for each template
        by_templ = self.df.groupby(['templ_dt','quality']).size()
        cnt_df = by_templ.to_frame(name='count').reset_index()
        # Don't want to print, it will be 4 lines for each template
        print(cnt_df.to_string(index=False))
        for tdt,row in cnt_df.iterrows():
            tidx = row['templ_dt']
            if not tidx in templ_stats:
                templ_stats[tidx] = [0] * 5
            qcount = templ_stats[tidx]
            q = int(row['quality'])
            qcount[q] = row['count']
            templ_stats[tidx] = qcount   # includes index 0 which is not used yet

        # average quality for each template
        by_templ = self.df.groupby('templ_dt')['quality'].mean()
        mean_df = by_templ.to_frame(name='mean').reset_index()
        for tdt,row in mean_df.iterrows():
            tidx = row['templ_dt']
            templ_stats[tidx][0] = row['mean']
        print(templ_stats)

    def family(self):
        '''
        Create DataFrame with template time and detection time.
        Column 1 is template_dt, Column 2 is a list of all associated det_dt
        :return: DF of families
        NOTE: there may be templates that detected other templates. These detections need to be
        consolidated into a single family.
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
        return bad_df['filename'].tolist()

    def exclude_self(self, time_diff_sec=30):
        '''
        Remove self-detection images: a template matches itself
        '''
        time_diff = dt.timedelta(seconds=time_diff_sec) # seconds
        print('exclude_self:')
        print('self.df has {} rows'.format(self.df.shape[0]))
        self.df = self.df[~self.df.apply(lambda x: dt_match(x['templ_dt'], x['det_dt'], time_diff), axis=1)]
        print('self.df has {} rows'.format(self.df.shape[0]))

    def save(self):
        if self.savefile:
            self.df.to_csv(self.match_file, index=False)
            print('MatchImages.save: updating file {}'.format(self.match_file))
        else:
            print('MatchImages.save: no update to {}'.format(self.match_file))

def getFamilyEvents(templates, family_df):
    '''
    Each template has 0 or more matches.
    For each template, append the list of match times.
    The returned DataFrame can be plotted on maps or used to construct eq family plots of time and 1D location.
    :param templates: instance of EqTemplates
    :param family_df: families found by families.Families
    :return: DataFrame with template hypocenter and list of detection times
    '''
    # Gather all templates that have matches and attach a list of each detection time
    # DataFrame with two columns: templ_dt (single datetime) and det_dt (list of detection datetimes)
    #family_df = matches.family()

    # Now lookup each template time to fetch the hypocenter info.
    # Each row is a unique template event. Add information about the detections to each row.
    # TODO: this is pretty inefficient
    templ_list = []     # it will be a list of lists; each element is a single template event + all new detections
    for index,row in family_df.iterrows():
        templ = templates.find(row['templ_dt'])     # lookup template event info for the current row
        if not templ.empty:
            templ_row = templ.iloc[0].tolist()      # convert event info to a list
            det_dt = []
            det_dt.extend(row['old_dt'])
            det_dt.extend(row['new_dt'])
            templ_row.append(len(det_dt))    # number of detected events
            templ_row.append(det_dt)         # append list of all matched detections
            templ_list.append(templ_row)
        else:
            # Unexpected that we will not match
            print('SKIP: no match for {}'.format(row['templ_dt']))
    # Each row in templ_list has the columns in the order shown - build a new DF
    df_new = pd.DataFrame(data=templ_list, columns=['time','longitude','latitude','depth','mag','region',
                                                    'template','templ_file','templ_dt','num_det','det_dt'])
    # tally number of detection matches
    match_count = df_new['num_det'].value_counts()
    print('getFamilyEvents: num_det =\n{}'.format(match_count))
    return df_new
