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

def getImageFilenames(path):
    print('getImageFilenames: {}'.format(path))
    files = os.listdir(path)
    print('getImageFilenames: len = {}'.format(len(files)))
    return sorted(files)

def openMatchFile(csvfile = MATCH_RECORD_FILE):
    lastline = None
    # if file already exist, get last line of file so we can restart where we left off
    if os.path.isfile(csvfile):
        with open(csvfile, 'r') as f:
            line = None
            for line in f:
                pass
            lastline = line
    else:
        # new file, write header
        with open(csvfile, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file','templ_date','detect_date','quality'])
    f = open(csvfile, 'a', newline='')
    writer = csv.writer(f)
    return f,writer,lastline

def storeMatchRecord(match_file, templ_date, det_date, quality):
    global csv_writer
    tstr = templ_date.strftime('%Y-%m-%d %H:%M:%S')
    dstr = det_date.strftime('%Y-%m-%d %H:%M:%S')
    csv_writer.writerow([match_file, tstr, dstr, quality])

def mapSetup(axes):
    # use low resolution coastlines.
    #map = Basemap(projection='merc',llcrnrlat=52, urcrnrlat=59, llcrnrlon=-162, urcrnrlon=-150, lat_ts=55, resolution='l', ax=axes)
    map = Basemap(projection='cyl',llcrnrlat=52, urcrnrlat=59, llcrnrlon=-162, urcrnrlon=-150, resolution='l', ax=axes)
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='coral',lake_color='aqua')
    # draw the edge of the map projection region (the projection limb)
    map.drawmapboundary(fill_color='aqua')
    # draw lat/lon grid lines every 30 degrees.
    map.drawmeridians(np.arange(200,210,1))
    map.drawparallels(np.arange(52,59,1))

    return map

# See https://www.pythonguis.com/tutorials/first-steps-qt-creator/
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setupHandlers()
        self.imageDir = IMAGE_DIR
        self.files = getImageFilenames(IMAGE_DIR)
        print('number of files = {}'.format(len(self.files)))
        self.fileIndex = -1
        self.imgName = None
        self.map = None
        self.templates = EqTemplates()
        self.makeMapPlot()

    def setupHandlers(self):
        self.nextButton.clicked.connect(self.clickNext)
        self.prevButton.clicked.connect(self.clickPrev)
        self.saveExitButton.clicked.connect(self.clickSaveExit)
        #self.saveGoodMatchButton.clicked.connect(self.clickSaveGoodMatch)
        self.saveGoodMatchButton.clicked.connect(lambda: self.clickMatch(4))
        self.saveMaybeMatchButton.clicked.connect(lambda: self.clickMatch(3))
        self.saveNotGoodMatchButton.clicked.connect(lambda: self.clickMatch(2))
        self.saveNoMatchButton.clicked.connect(lambda: self.clickMatch(1))

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
        # plot
        self.map = mapSetup(self.map_axes)

    def mapMarker(self, location):
        '''
        Display a marker on the map. This redraws the entire map so that previous markers are not displayed.
        :param location: tuple of (longitude, latitude, depth)
        :return:
        '''
        #self.plotWidget.figure.clf()
        #self.plotWidget = None
        self.map_axes.clear()
        self.makeMapPlot(self.map_axes)
        x,y = self.map(location[0], location[1])
        self.locationLabel.setText('{:.4f}, {:.4f}'.format(location[0], location[1]))
        self.map.plot(x, y, 'bo', markersize=4.0)
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
        self.storeMatch(value)
        self.clickNext()

    def storeMatch(self,quality):
        storeMatchRecord(self.imgName, self.templ_date, self.det_date, quality)

    def showDates(self, templ_date, det_date):
        # show the template and detection dates
        self.templateDateLabel.setText(templ_date.strftime('%Y-%m-%d %H:%M:%S'))
        self.matchDateLabel.setText(det_date.strftime('%Y-%m-%d %H:%M:%S'))

    def changeImage(self, imgName):
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
        matches = self.templates.find(self.templ_date)
        print(matches.iloc[0])
        ev = matches.iloc[0]
        # display the template location on a map
        location = (ev['longitude'], ev['latitude'], ev['depth'])
        self.mapMarker(location)

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
        self.imgName = self.files[self.fileIndex]
        self.changeImage(self.imgName)

    def showImage(self, image_name):
        image_profile = QtGui.QImage(image_name)
        img_geom = self.image.geometry()
        image_profile = image_profile.scaled(img_geom.width(),img_geom.height(), aspectRatioMode=QtCore.Qt.KeepAspectRatio, transformMode=QtCore.Qt.SmoothTransformation) # To scale image for example and keep its Aspect Ration
        self.image.setPixmap(QtGui.QPixmap.fromImage(image_profile))

class MatchRecords:
    # keep a list of all match images that we have reviewed
    def __init__(self, csvfile):
        if os.path.isfile(csvfile):
            self.fp, self.writer, self.lastline = self.openMatchFile(csvfile)
        else:
            self.fp, self.writer, self.lastline = self.createMatchFile(csvfile)

    def createMatchFile(self, csvfile):
        # new file, write header
        lastline = None
        f = open(csvfile, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['file','templ_date','detect_date','quality'])
        return f,writer,lastline

    def openMatchFile(self, csvfile = MATCH_RECORD_FILE):
        lastline = None
        # if file already exist, get last line of file so we can restart where we left off
        with open(csvfile, 'r') as f:
            line = None
            for line in f:
                pass
            lastline = line
        f = open(csvfile, 'a', newline='')
        writer = csv.writer(f)
        return f,writer,lastline

    def storeMatchRecord(self, match_file, templ_date, det_date, quality):
        tstr = templ_date.strftime('%Y-%m-%d %H:%M:%S')
        dstr = det_date.strftime('%Y-%m-%d %H:%M:%S')
        self.writer.writerow([match_file, tstr, dstr, quality])

    def close(self):
        self.fp.close()

class EqTemplates:
    # maintain a list of all templates and enable lookup of their event data
    time_diff = dt.timedelta(seconds=30)

    def __init__(self):
        self.templ_csv = 'ev_selected_region.csv'
        # open the file with pandas
        # it looks like this
        # time,longitude,latitude,depth,mag,region,template,templ_file
        # 2018-05-12T08:56:33.501000Z,-161.5485,54.3787,25500.0,2.2,0,2018-05-12T08:56:33.000000Z,2018_05_12t08_56_33.tgz
        self.df = pd.read_csv(self.templ_csv)
        # convert 'time' to UTCDateTime
        self.df['time'] = self.df['time'].apply(lambda tstr: dt.datetime.strptime(tstr, '%Y-%m-%dT%H:%M:%S.%fZ'))
        #self.df['time'] = self.df['time'].replace(lambda tstr: dt.datetime.strptime(tstr, '%Y-%m-%dT%H:%M:%S.%fZ'))
        #self.df['date'] = self.df['time'].apply(lambda t: t.date())
        print(self.df)
        #self.templates = df

    def find(self, templ_t, dt_format='%Y-%m-%dT%H-%M-%S'):
        # templ_time is from the filename of a match_filter PNG image
        # PNG filename: Templ-2018-05-19T08-14-07_Det-2018-05-22T05-24-26.png
        # make approximate match. Within 30 seconds?
        # return the data from pandas with lat-long and region number
        #templ_t = dt.datetime.strptime(templ_time, dt_format)
        #new_df = self.df[self.df.apply(lambda x: abs(templ_t - x['time']) < self.time_diff)]
        new_df = self.df[self.df['time'].apply(lambda x: abs(templ_t - x) < self.time_diff)]
        #print(new_df)
        return new_df

if __name__ == "__main__":
    csvfile,csv_writer,lastLine = openMatchFile()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    if lastLine:
        last = lastLine.split(',')[0]
    else:
        last = None
    window.setLast(last)
    window.show()
    retval = app.exec_()
    print('Exiting: {}'.format(retval))
    csvfile.close()
    sys.exit(retval)
