# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import os
from PyQt5 import QtCore,QtGui, QtWidgets
import matplotlib
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QVBoxLayout,QHBoxLayout,QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#import cartopy.crs as ccrs
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes,zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from qt5_match import Ui_MainWindow
#from obspy import UTCDateTime
import datetime as dt
import numpy as np
import pandas as pd
import csv
csv_writer = None

IMAGE_DIR = '/proj/shumagin/gnelson/plots/detections'
MATCH_RECORD_FILE = '/proj/shumagin/gnelson/match_records.csv'
IMAGE_DIR = 'E:\Glenn Nelson\science\eq_gaps\plots\detections'
MATCH_RECORD_FILE = 'E:\Glenn Nelson\science\eq_gaps\match_records.csv'
IMAGE_DIR = '.\detections'
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
    :param dt1:
    :param dt2:
    :param dt_diff: datetime.timedelta value
    :return: True if match
    '''
    return abs(dt1 - dt2) < dt_diff

class MatchImages:
    '''
    Images have filenames like this: Templ-2018-11-17T03-53-30_Det-2018-05-08T14-24-02.png
    Contains df (Pandas DataFrame) with templ_dt and region as columns.
    Method _findRegion will lookup the region by searching EqTemplates object.
    '''
    def __init__(self, path, templates):
        self.imageDir = path
        self.templates = templates
        # TODO: keep full list always, keep separate filtered list
        # TODO: regionLookup is dumb, should just merge with allFiles
        self.regionLookup = {}
        df = pd.DataFrame(columns=['filename','templ_dt','det_dt','region'])
        self.allFiles = None
        self.notSeenFiles = None
        self.files = None   # this is the current working set after all filters
        files = []
        with os.scandir(self.imageDir) as ls:
            for item in ls:
                if item.is_file():
                    filename = str(item.name)
                    files.append(filename)
        print('MatchImages: len = {}'.format(len(files)))
        self.allFiles = sorted(files)   # it will sort on template date
        self.files = self.allFiles
        rows = []
        for file in self.allFiles:
            templ_dt, det_dt = self.parseImageFilename(file)
            f_dict = {'filename':file, 'templ_dt':templ_dt, 'det_dt':det_dt, 'region':'-'}
            rows.append(f_dict)
        self.df = df.append(rows, ignore_index=True)
        self.df = self.df.sort_values(by=['templ_dt'])
        #print(self.df)
        self._findRegions(templates)

    @classmethod
    def parseImageFilename(cls, imgName):
        # extract template date and detection date
        f,x = imgName.split('.')
        t,d = f.split('_')
        templ_date = dt.datetime.strptime(t[len('Templ-'):], '%Y-%m-%dT%H-%M-%S')
        det_date = dt.datetime.strptime(d[len('Det-'):], '%Y-%m-%dT%H-%M-%S')
        return templ_date,det_date

    @timing
    def _findRegions(self, templates):
        # Match all image filenames with assigned region for the template
        # compare columns in 2 DF for matches: https://datascience.stackexchange.com/questions/33053/how-do-i-compare-columns-in-different-data-frames
        print(self.df)
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
                    #print('new match at {}, dt={}'.format(cnt,templ_dt))
                tdt = templ_dt
                if not match.empty:
                    region = match.iloc[0]['region']
                    if not region in self.regionLookup:
                        self.regionLookup[region] = list()
                    self.regionLookup[region].append(row['filename'])
            except:
                print('ERROR in _findRegions: file={}'.format(row['filename']))

    # My original attempt is painfully slow.
    @timing
    def _findRegions0(self, templates):
        # Match all image filenames with assigned region for the template
        cnt = 1
        # TODO: use self.df
        for file in self.allFiles:
            if cnt % 250 == 0:
                print('findRegions: {}'.format(cnt))
            cnt += 1
            try:
                templ_dt, det_dt = self.parseImageFilename(file)
                match = templates.find(templ_dt)  # return a DF of record from EV_SELECTED_REGION
                if not match.empty:
                    region = match.iloc[0]['region']
                    if not region in self.regionLookup:
                        self.regionLookup[region] = list()
                    self.regionLookup[region].append(file)
            except:
                print('ERROR in _findRegions: file={}'.format(file))

    def getImageDir(self):
        return self.imageDir

    def removeSeen(self, seenFiles):
        '''
        Remove any image files from list if they've already been seen.
        Modifies self.files, leaves self.allFiles intact.
        :param seenFiles: list of filenames from MatchRecords
        :return:
        '''
        #print('removeSeen begin {}, remove {}'.format(len(self.allFiles), len(seenFiles)))
        notSeen = set(self.allFiles) - set(seenFiles)
        self.notSeenFiles = sorted(list(notSeen))
        self.files = self.notSeenFiles
        #print('removeSeen end {}'.format(len(self.files)))

    def filterRegion(self, region='All'):
        '''
        Remove image files from list unless they match the region parameter.
        Region parameter is found in EqTemplates.
        :param region: 'All' or other values, '0' to '7' for Shumagin study
        :return:
        '''
        # TODO: not finished
        # start with allFiles, reapply removeSeen, and finally select only the images from region
        #self.removeSeen()
        if region == 'All':
            # TODO: maybe not use AllFiles
            self.files = self.notSeenFiles
        else:
            self.files = []
            if region in self.regionLookup:
                self.files = self.regionLookup[region]
        print('filterRegion: {} has {} images'.format(region, len(self.files)))
        return self.files

    def getFiles(self):
        return self.files

def mapSetup(axes, bbox):
    '''
    Setup a map object, it's an instance of Basemap
    :param axes: returned from Figure
    :param bbox: tuple(tuple(lower-left long,lat), tuple(upper-right long, lat))
    :return: basemap
    '''
    # use low resolution coastlines.
    #map = Basemap(projection='merc',llcrnrlat=52, urcrnrlat=59, llcrnrlon=-162, urcrnrlon=-150, lat_ts=55, resolution='l', ax=axes)
    map = Basemap(projection='cyl',llcrnrlat=bbox[0][1], urcrnrlat=bbox[1][1],
                  llcrnrlon=bbox[0][0], urcrnrlon=bbox[1][0], resolution='l', ax=axes)
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='coral',lake_color='aqua')
    # draw the edge of the map projection region (the projection limb)
    map.drawmapboundary(fill_color='aqua')
    # draw lat/lon grid lines every 1 degrees.
    map.drawmeridians(np.arange(200,210,1))
    map.drawparallels(np.arange(52,59,1))
    return map

# See https://www.pythonguis.com/tutorials/first-steps-qt-creator/
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, matchRecords, matchImages, stations, templates, **kwargs):
        super(MainWindow, self).__init__(**kwargs)
        self.setupUi(self)
        # must addItems before setupHandlers
        self.regionSelect.addItems(['All','0','1','2','3','4','5','6','7'])
        self.setupHandlers()
        #self.imageDir = IMAGE_DIR
        self.files = matchImages.getFiles()
        print('MainWindow.init: files = {}'.format(len(self.files)))
        self.templates = templates
        self.matchRecords = matchRecords
        self.matchImages = matchImages
        self.stations = stations
        print('number of files = {}'.format(len(self.files)))
        self.fileIndex = -1
        # TODO: imageDir should not be here, we should be retrieving images from MatchImages class
        self.imageDir = matchImages.getImageDir()
        self.imgName = None
        self.map = None
        self.map_axes = None
        self.map_inset_axes = None
        print('call makeMapPlot')
        self.makeMapPlot()
        self.clickNext()

    def setupHandlers(self):
        self.nextButton.clicked.connect(self.clickNext)
        self.prevButton.clicked.connect(self.clickPrev)
        self.saveExitButton.clicked.connect(self.clickSaveExit)
        #self.saveGoodMatchButton.clicked.connect(self.clickSaveGoodMatch)
        self.saveGoodMatchButton.clicked.connect(lambda: self.clickMatch(4))
        self.saveMaybeMatchButton.clicked.connect(lambda: self.clickMatch(3))
        self.saveNotGoodMatchButton.clicked.connect(lambda: self.clickMatch(2))
        self.saveNoMatchButton.clicked.connect(lambda: self.clickMatch(1))
        self.regionSelect.currentTextChanged.connect(self.changeRegion)

    def changeRegion(self):
        region = self.regionSelect.currentText()
        print('new region {}'.format(region))
        #templ_df = self.templates.regionSelect(region)
        self.files = self.matchImages.filterRegion(region)
        self.fileIndex = -1
        self.clickNext()

    def makeMapPlot(self, ax1=None):
        '''
        Map of the region. Displayed in a QtWidget
        :param ax1: matplotlib Axes object from a Figure
        :return:
        '''
        if not ax1:
            fig, ax1 = plt.subplots()
            #print(ax1)
            self.plotWidget = FigureCanvas(fig)
            self.map_axes = ax1
            lay = QVBoxLayout()
            #lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.plotWidget)
            self.mapWidget.setLayout(lay)
            #self.map_inset_axes = zoomed_inset_axes(self.map_axes, 6, loc=4)
            self.map_inset_axes = inset_axes(self.map_axes, width='50%', height='50%', loc='lower right')
            # xlim, ylim will change and must be set in mapMarker method
            #self.map_inset_axes.set_xlim()
            #self.map_inset_axes.set_ylim()
        # plot map
        # llcrnrlat=52, urcrnrlat=59, llcrnrlon=-162, urcrnrlon=-150
        bound_box = ((-162.0, 52.0),(-150.0, 59.0))
        self.map = mapSetup(self.map_axes, bound_box)

    def mapMarker(self, location):
        '''
        Display a marker on the map. This redraws the entire map so that previous markers are not displayed.
        :param location: tuple of (longitude, latitude, depth)
        :return:
        '''
        #self.plotWidget.figure.clf()
        #self.plotWidget = None
        self.map_axes.clear()
        if self.map_inset_axes:
            self.map_inset_axes.clear()
        self.makeMapPlot(self.map_axes)
        # plot the stations first
        lons = self.stations.df['longitude'].tolist()
        lats = self.stations.df['latitude'].tolist()
        sta_names = self.stations.df['station'].tolist()
        sta_x,sta_y = self.map(lons, lats)
        if False:
            # nice idea, but you can't set the markersize!
            self.map.scatter(sta_x, sta_y, color='red', marker='^')
        else:
            for lon,lat in zip(sta_x,sta_y):
                self.map.plot(lon, lat, color='red', marker='^', markersize=2.0)
        # Now plot the event
        ev_x,ev_y = self.map(location[0], location[1])
        self.locationValue.setText('{:.4f}, {:.4f}'.format(location[0], location[1]))
        self.map.plot(ev_x, ev_y, 'bo', markersize=4.0)

        # map inset to zoom into location
        lon0 = location[0] - 1.0
        lon1 = location[0] + 1.0
        lat0 = location[1] - 1.0
        lat1 = location[1] + 1.0
        self.map_inset_axes.set_xlim(lon0, lon1)
        self.map_inset_axes.set_ylim(lat0, lat1)
        map2 = Basemap(llcrnrlon=lon0, urcrnrlon=lon1,
                       llcrnrlat=lat0, urcrnrlat=lat1,
                       projection='cyl', resolution='l', ax=self.map_inset_axes)
        # draw coastlines, country boundaries, fill continents.
        map2.drawcoastlines(linewidth=0.25)
        map2.drawcountries(linewidth=0.25)
        map2.fillcontinents(color='coral',lake_color='aqua')
        # draw the edge of the map projection region (the projection limb)
        map2.drawmapboundary(fill_color='aqua')
        #map2.drawmapboundary(fill_color='#7777ff')
        #map2.fillcontinents(color='#ddaa66', lake_color='#7777ff', zorder=0)
        #map2.drawcoastlines()
        #map2.drawcountries()
        # plot stations with labels
        for lon,lat,name in zip(sta_x, sta_y, sta_names):
            #map2.plot(lon, lat, color='red', marker='^')
            # kwargs are passed to Text object
            self.map_inset_axes.annotate(name, (lon, lat), color='red', fontsize='xx-small')
        map2.plot(ev_x, ev_y, 'bo', markersize=4.0)
        self.plotWidget.figure.canvas.draw()    # this refreshes the map and adds new marker

    def makeTestPlot(self):
        data = np.array([0.7, 0.7, 0.7, 0.8, 0.9, 0.9, 1.5, 1.5, 1.5, 1.5])
        fig, ax1 = plt.subplots()
        bins = np.arange(0.6, 1.62, 0.02)
        n1, bins1, patches1 = ax1.hist(data, bins, alpha=0.6, density=False, cumulative=False)
        # plot
        self.plotWidget = FigureCanvas(fig)
        lay = QVBoxLayout()
        #lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plotWidget)
        self.mapWidget.setLayout(lay)

    def setLast(self,lastfile):
        '''
        Start with last event viewed, if any.
        Find lastfile in the list of all files, set the index and bring up display of next image.
        :param lastfile: filename of a template match PNG file.
        :return:
        '''
        if lastfile:
            idx = self.files.index(lastfile)
            if idx >= 0 and idx < len(self.files):
                self.fileIndex = idx
            else:
                print('ERROR: last={}, but not found in list'.format(lastfile))
        self.clickNext()

    def clickSaveExit(self):
        '''
        store the last image we looked at, so we can resume later
        :return:
        '''
        # close the MainWindow, thereby exiting the app and closing the csvfile
        self.close()

    def clickMatch(self, value):
        # handler for any of the match quality buttons
        self.storeMatch(self.region, value, self.is_new)
        self.clickNext()

    def storeMatch(self, region, quality, is_new):
        self.matchRecords.saveMatch(self.imgName, self.templ_date, self.det_date, region, quality, is_new)

    def showDates(self, templ_date, det_date):
        # show the template and detection dates
        self.templateDateLabel.setText(templ_date.strftime('%Y-%m-%d %H:%M:%S'))
        self.matchDateLabel.setText(det_date.strftime('%Y-%m-%d %H:%M:%S'))

    def changeImage(self, imgName):
        print('changeImage: {}'.format(imgName))
        self.filenameLabel.setText(imgName) # display the image filename
        # extract template date and detection date
        f,x = imgName.split('.')
        t,d = f.split('_')
        self.templ_date = dt.datetime.strptime(t[len('Templ-'):], '%Y-%m-%dT%H-%M-%S')
        self.det_date = dt.datetime.strptime(d[len('Det-'):], '%Y-%m-%dT%H-%M-%S')
        self.showDates(self.templ_date, self.det_date)
        # display the template match traces
        self.showImage('{}/{}'.format(self.imageDir,imgName))
        # lookup the template to get its geographic location
        templ_data = self.templates.find(self.templ_date)
        #print(templ_data.iloc[0])
        ev = templ_data.iloc[0]
        detect_data = self.templates.find(self.det_date)
        if not detect_data.empty:
            # detection event is also a template
            self.is_new = 0
            print('detection {} is also a template'.format(self.det_date))
            self.inCatalogValue.setText('event is a template')
            self.inCatalogValue.setStyleSheet('color: red')
            if detect_data.shape[0] > 1:
                print('WARNING: detection matches more than one template?')
                for i in range(detect_data.count()):
                    print('... {}'.format(detect_data.iloc[i]))
        else:
            self.is_new = 1
            print('detection {} is not a template'.format(self.det_date))
            self.inCatalogValue.setText('event is new')
            self.inCatalogValue.setStyleSheet('color: green')
        # display the template location on a map
        location = (ev['longitude'], ev['latitude'], ev['depth'])
        self.region = str(ev['region'])
        self.evRegionValue.setText('{}'.format(region))
        self.magnitudeValue.setText('{:.1f}'.format(ev['mag']))
        self.mapMarker(location)
        print('changeImage: END')

    def clickPrev(self):
        self.fileIndex -= 1
        if self.fileIndex < 0:
            self.fileIndex = 0
        self.imgName = self.files[self.fileIndex]
        self.changeImage(self.imgName)

    def clickNext(self):
        self.fileIndex += 1
        if self.fileIndex > len(self.files):
            self.fileIndex -= 1
        print('clickNext: index={}'.format(self.fileIndex))
        print('clickNext: files len={}'.format(len(self.files)))
        print('clickNext: type(files): {}'.format(type(self.files)))
        self.imgName = self.files[self.fileIndex]
        print('clickNext: imgName={}'.format(self.imgName))
        self.changeImage(self.imgName)

    def showImage(self, image_name):
        image_profile = QtGui.QImage(image_name)
        img_geom = self.image.geometry()
        image_profile = image_profile.scaled(img_geom.width(),img_geom.height(),
                                             aspectRatioMode=QtCore.Qt.KeepAspectRatio,
                                             transformMode=QtCore.Qt.SmoothTransformation) # To scale image for example and keep its Aspect Ration
        self.image.setPixmap(QtGui.QPixmap.fromImage(image_profile))

class Stations:
    # all station locations for plotting on maps
    def __init__(self, station_file=STATION_FILE):
        self.df = pd.read_csv(station_file)

class MatchRecords:
    # keep a list of all match images that we have reviewed
    # TODO: change this to use DictWriter or Pandas
    def __init__(self, csvfile = MATCH_RECORD_FILE):
        self.recordFile = csvfile
        self.df = None
        self.matchFiles = []    # list of all filenames we've reviewed
        if os.path.isfile(csvfile):
            #self.fp, self.writer = self.openMatchFile(csvfile)
            self.readFile()
        else:
            self.createDF()

    def getFilenames(self):
        return self.matchFiles

    def createMatchFile(self, csvfile):
        # new file, write header
        f = open(csvfile, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['file','templ_date','detect_date','quality','is_new'])
        return f,writer

    def createDF(self):
        self.df = pd.DataFrame(columns=['file','templ_date','detect_date','region','quality','is_new'])

    def readFile(self, csvfile):
        self.df = pd.read_csv(csvfile)
        self.matchFiles = self.df['file'].tolist()

    def saveMatch(self, match_file, templ_date, det_date, region, quality, is_new):
        '''
        Stores our evaluation of the match quality in file.
        Also adds the match_file name to internal list so that we know we've seen it during this session.
        :param match_file: filename of match image file
        :param templ_date: date of the template (also embedded in the filename)
        :param det_date: date of the detection (also embedded in the filename)
        :param region: designation for the region that the template is located in
        :param quality: evaluated quality of detection, a number from 1 to 4
        :param is_new: False: if detection is another template that we already have.
        :return:
        '''
        print('saveMatch: {},{},{},{},{},{}'.format(match_file,templ_date,det_date,region,quality,is_new))
        tstr = templ_date.strftime('%Y-%m-%d %H:%M:%S')
        dstr = det_date.strftime('%Y-%m-%d %H:%M:%S')
        match = {'file':match_file,'templ_date':tstr,'detect_date':dstr,'quality':quality,'is_new':is_new,'region':region}
        #self.df = self.df.append([match_file,tstr,dstr,quality,is_new])
        #self.df = self.df.append(match)
        print(match)
        newdf = pd.DataFrame([match,])
        print(newdf)
        self.df = pd.concat([self.df, newdf])
        print('saveMatch: END')

    def openMatchFile(self, csvfile = MATCH_RECORD_FILE):
        nline = 0
        # if file already exist, get last line of file so we can restart where we left off
        with open(csvfile, 'r') as f:
            line = None
            for line in f:
                if nline > 0:
                    fname,_ = line.split(',',1)
                    self.matchFiles.append(fname)
                nline += 1
        f = open(csvfile, 'a', newline='')
        writer = csv.writer(f)
        return f,writer

    def storeMatchRecord(self, match_file, templ_date, det_date, quality, is_new):
        '''
        Stores our evaluation of the match quality in file.
        Also adds the match_file name to internal list so that we know we've seen it during this session.
        :param match_file: filename of match image file
        :param templ_date: date of the template (also embedded in the filename)
        :param det_date: date of the detection (also embedded in the filename)
        :param quality: evaluated quality of detection, a number from 1 to 4
        :param is_new: False: if detection is another template that we already have.
        :return:
        '''
        self.matchFiles.append(match_file)
        tstr = templ_date.strftime('%Y-%m-%d %H:%M:%S')
        dstr = det_date.strftime('%Y-%m-%d %H:%M:%S')
        self.writer.writerow([match_file, tstr, dstr, quality, is_new])

    def close(self):
        self.df.to_csv(self.recordFile, index=False)

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
        # convert 'time' to UTCDateTime
        #self.df['time'] = self.df['time'].apply(lambda tstr: dt.datetime.strptime(tstr, '%Y-%m-%dT%H:%M:%S.%fZ'))
        df['templ_dt'] = df['time'].apply(lambda tstr: dt.datetime.strptime(tstr[:19], '%Y-%m-%dT%H:%M:%S'))
        df['depth'] = df['depth'].div(1000.0)
        self.df = df.sort_values(by=['templ_dt'])
        print(self.df)
        #self.templates = df

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

if __name__ == "__main__":
    #csvfile,csv_writer,lastLine = openMatchFile()
    matchRecords = MatchRecords(MATCH_RECORD_FILE)
    stations = Stations()
    templates = EqTemplates()
    matchImages = MatchImages(IMAGE_DIR, templates)
    matchImages.removeSeen(matchRecords.getFilenames())
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(matchRecords, matchImages, stations, templates)
    window.show()
    retval = app.exec_()
    print('Exiting: {}'.format(retval))
    matchRecords.close()
    sys.exit(retval)
