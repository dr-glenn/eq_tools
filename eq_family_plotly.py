# Read match_records.csv and generate lists of template families.
# The families will be plotted with time on X and location along strike (trench) on Y.
# The template event is a different symbol from the matches found by EqCorrScan, but the events are connected by lines.
'''
EqCorrscan can generate template families when performing match_filter detection. However we are looking
over a time period of a year and therefore the builtin family methods are not useful.
In program my_detect.py we load a batch of templates from a geographic zone (creating a Tribe)
and run match_filter (actually Tribe.detect) over the span of one day. my_detect.py continues to
advance over the calendar, one day at a time.
The file match_records.csv pairs a template with a detection: they are uniquely identified by datetime.
We gather all rows of the same template and create a family of all the new detections.

In order to plot templates according to distance along strike, use the 'cross track' calculation as given here:
https://www.movable-type.co.uk/scripts/latlong.html
'''
import os
import sys
import datetime as dt
import numpy as np
import argparse
import plotly.express as px
import pandas as pd
import haversine as hv
from templates import EqTemplates
from matches import MatchImages, getFamilyEvents
import config as cfg
from config import MATCH_RECORD_FILE,STATION_FILE,EV_REGION_FILE,DETECTION_PLOTS
from my_util import dt_match

IMAGE_DIR = DETECTION_PLOTS
TEMPLATE_POLY = [ (-160.5,52.5), (-162.5,54.8), (-152.0,58.5), (-149.5,56.6), (-160.5,52.5) ]

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

def zone_bounds(ll0, ll1, nzone):
    a12,c12,d12,bearing12 = hv.haversine(ll0, ll1)
    print('region length = {} km'.format(d12))
    dzone = d12 / nzone
    # midpoints
    #zones = [dzone * (float(i) + 0.5) for i in range(0,nzone)]
    #endpoints
    zones = [dzone * float(i) for i in range(0,nzone)]
    return zones

# TODO: should plot line segments that also connect templates together in the event that a match is not a new event
def plot_matches(fam_df, fig, legend_colors, num_min=3, yaxis='longitude'):
    '''
    Using family_df, add lines to template location plot where each line shows new event detections
    with datetime on X-axis and distance along strike on Y-axis.
    Since we don't have locations for these match detections, we use same location as template.
    Each detection time is marked with a symbol that is different from the template event.
    :param fam_df: the DF with families of events
    :param fig: plotly figure, this function adds to it
    :param legend_colors: list of discrete colors in the legend of the original plot
    :param num_min: minimum number of new detected events to plot (so that the figure isn't too cluttered)
    :param yaxis: column name in fam_df for y-axis. 'longitude' or 'dist' are supported.
    :return:
    '''
    df = fam_df[['templ_dt',yaxis,'det_dt','num_det']].copy()
    df['inum_det'] = pd.to_numeric(df['num_det'])
    new_df = df[df['inum_det'] >= num_min]
    for index,row in new_df.iterrows():
        inum = int(row['inum_det'])
        yval = [row[yaxis]] * inum
        color = legend_colors[inum-1]
        #print(yval)
        fig.add_scatter(x=row['det_dt'], y=yval, mode='lines+markers', marker_symbol='diamond-open', marker_size=10,
                        line=dict(color=color), showlegend=False)

# TODO: attempt to plot scatter lines with different symbol for template_dt and det_dt. Not working.
def plot_matches1(fam_df, fig, legend_colors, num_min=3):
    '''
    Using family_df, add lines to template location plot where each line shows new event detections
    with datetime on X-axis and distance along strike on Y-axis.
    Since we don't have locations for these match detections, we use same location as template.
    Each detection time is marked with a symbol that is different from the template event.
    :param fam_df: the DF with families of events
    :param fig: plotly figure, this function adds to it
    :param legend_colors: list of discrete colors in the legend of the original plot
    :param num_min: minimum number of new detected events to plot (so that the figure isn't too cluttered)
    :return:
    '''
    df = fam_df[['templ_dt','longitude','det_dt','num_det']].copy()
    df['inum_det'] = pd.to_numeric(df['num_det'])
    new_df = df[df['inum_det'] >= num_min]
    for index,row in new_df.iterrows():
        inum = int(row['inum_det'])
        color = legend_colors[inum-1]
        xval = pd.Series(row['templ_dt'])
        xvals = pd.concat([xval, pd.Series(row['det_dt'])])
        yvals = [row['longitude']] * (inum+1)
        marker = pd.Series('circle')
        markers = pd.concat([marker,pd.Series(['diamond-open',] * inum)])
        print(markers)
        fig.add_scatter(x=xvals, y=yvals, mode='lines+markers', symbol=markers, marker_size=10,
                        line=dict(color=color), showlegend=False)

def plot_zones(fig, xval, zones):
    '''
    Plot markers along Y-axis to designate the zones within our study region
    :param fig:
    :param xval:
    :param zones:
    :return:
    '''
    x = [xval] * (len(zones)-1)
    fig.add_scatter(x=x, y=zones[1:], mode='markers', marker_color='black', marker_size=16, marker_symbol='triangle-right', showlegend=False)

# use plotly
def plot(df, num_min, title, yaxis='longitude'):
    '''
    Plot all templates in df with datetime on X-axis, location along strike on Y-axis.
    This function only plots template events and color codes them for number of matches found.
    (NOTE: currently using longitude as proxy for distance along strike)
    :param df:
    :param num_min: minimum family size to display as timelines
    :param title: title of plot
    :param yaxis: 'longitude' or 'dist'. If 'dist', then mark zone boundaries on yaxis
    :return:
    '''
    '''
    Use Plotly discrete color scale: https://plotly.com/python/discrete-color/
    There are 10 values (indexed from 0 to 9) available.
    Values are offset to fit in range 0 to 9, but I don't think they're scaled.
    I am coloring the plot by number of matches for each template event, so the max should be 10.
    '''
    # num_det are int values - but then you get a continuous color spectrum - so convert to str
    max_det = df['num_det'].max()
    df['num_det'] = df['num_det'].apply(lambda x: str(int(x)))
    legend_vals = [str(i) for i in range(1,max_det+1)]
    print('legend: {}'.format(legend_vals))
    legend_colors = px.colors.qualitative.Plotly    # a list of RGB values
    # plot the template events as filled circles
    fig = px.scatter(df, x='templ_dt', y=yaxis, color='num_det', title=title,
                     category_orders={'num_det': legend_vals}, hover_data=['latitude','longitude'])
    fig.update_traces(marker={'size':12})
    # plot the families that are affiliated with a template
    plot_matches(df, fig, legend_colors, num_min=num_min, yaxis=yaxis)

    # plot the zone boundaries on the Y axis
    if yaxis == 'dist':
        endpt0 = np.asarray(TEMPLATE_POLY[0])
        endpt3 = np.asarray(TEMPLATE_POLY[3])
        zones = zone_bounds(endpt0, endpt3, 8)
        print('Zone boundaries: {}'.format(zones))
        xval = df['templ_dt'].min()     # datetime
        # Want to plot zone markers at edge of y-axis. Unfortuntely y-axis is automatically scaled and plotly
        # doesn't allow us to know the autoscale limits!
        # So assume a few days earlier is OK.
        #xval = xval - dt.timedelta(days=3)
        xval = dt.datetime(year=xval.year, month=xval.month, day=1)
        plot_zones(fig, xval, zones)

    fig.show()

def getDetectionFiles():
    '''
    Read the region detection files, match with template times and correct number of stations
    in each match.
    :return: list of DataFrames for each file 'detections_R?.csv'
    '''
    '''
    Files look like this:
    template_name,detect_time,no_chans,detect_val,detect_ratio,chans
    2019_04_27t01_41_43,2018-05-04T12:18:09.633400Z,2,1.33368,0.666838109493,"('CHI', 'BHZ'), ('S14K', 'BHZ')"
    '''
    detect_df = []
    for region in range(0,8):
        df = pd.read_csv(cfg.FILES_DIR+'detections_R{}.csv'.format(region))
        df['template_name'] = df['template_name'].apply(lambda x: dt.datetime.strptime(x, '%Y_%m_%dt%H_%M_%S'))
        df['detect_time']   = df['detect_time'].apply(lambda x: dt.datetime.strptime(x[:19], '%Y-%m-%dT%H:%M:%S'))
        for index,row in df.iterrows():
            chans = list(set(eval(row['chans'])))   # eliminate duplicate station picks
            df.at[index,'chans'] = chans
            df.at[index,'no_chans'] = len(chans)    # update in case there were duplicates
        detect_df.append(df.copy())
    return detect_df

def attachStationCounts(family_df, detect_dfs):
    #self.df = self.df[~self.df.apply(lambda x: dt_match(x['templ_dt'], x['det_dt'], time_diff), axis=1)]
    for index,row in family_df.iterrows():
        n_sta = []
        tdt = row['templ_dt']
        region = int(row['region'])
        df = detect_dfs[region][detect_dfs[region]['template_name'] == tdt]
        #print(df)
        det_dts = row['det_dt']
        #print('templ_dt={}, det_dts: {}'.format(tdt, det_dts))
        for det_dt in det_dts:
            match_df = df[df['detect_time'] == det_dt]
            if not match_df.empty:
                n = df[df['detect_time'] == det_dt].iloc[0].no_chans
                #print('n = {}'.format(n))
                n_sta.append(n)
            else:
                print('match_df is empty: det_dt = {}'.format(det_dt))
        #row['n_sta'] = n_sta
        if len(n_sta) > 0:
            family_df.loc[index, 'n_sta'] = str(n_sta)
        #if index==1: print(row)
    print(family_df)

def locationInBox(df):
    '''
    df has (longitude, latitude) of each event. Calculate position within study region bounds.
    :param df: DataFrame is modified with 2 new columns.
    :return:
    '''
    pt0 = np.asarray(TEMPLATE_POLY[0])
    pt1 = np.asarray(TEMPLATE_POLY[3])
    for index,row in df.iterrows():
        dist, offset = hv.line_dist(pt0, pt1, np.asarray((row['longitude'], row['latitude'])))
        df.at[index,'dist'] = dist
        df.at[index,'offset'] = offset

def main(quality, is_new, num_min=3):
    new_ev = 1 in is_new
    old_ev = 0 in is_new
    title = 'Quality={}, New Matches={}, Template Matches={}, Min Family={}'.format(quality, new_ev, old_ev, num_min)
    templates = EqTemplates()
    matches = MatchImages(IMAGE_DIR, templates)
    matches.filter(quality=quality, is_new=is_new)
    family_df = getFamilyEvents(templates, matches)
    family_df['n_sta']  = None
    family_df['dist']   = None
    family_df['offset'] = None
    locationInBox(family_df)
    print(family_df)
    print('dist: {} to {}'.format(family_df['dist'].min(), family_df['dist'].max()))
    print('offset: {} to {}'.format(family_df['offset'].min(), family_df['offset'].max()))
    det_dfs = getDetectionFiles()
    attachStationCounts(family_df, det_dfs)
    #print(family_df)
    plot(family_df, num_min, title, yaxis='dist')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display time plot of template families',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('-h','--help')
    parser.add_argument('-n', '--num_detect', dest='num_min', type=int, default=3, help='minimum number of detections in family')
    parser.add_argument('--include_template_matches', dest='include_template_matches', default=False, action='store_true',
                        help='include matches between templates, else only show new detections')
    parser.add_argument('--quality', dest='quality', type=str, default='3,4', help='match quality levels to display (1,2,3,4)')
    args = parser.parse_args()
    if args.include_template_matches:
        is_new = [0,1]
    else:
        is_new = [1,]
    quality = [int(x) for x in args.quality.split(',')]
    main(quality, is_new, num_min=args.num_min)