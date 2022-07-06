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
#import numpy as np
import argparse
import plotly.express as px
import pandas as pd
import math
from templates import EqTemplates
from matches import MatchImages, getFamilyEvents
import config as cfg
from config import MATCH_RECORD_FILE,STATION_FILE,EV_REGION_FILE,DETECTION_PLOTS
from my_util import dt_match

IMAGE_DIR = DETECTION_PLOTS

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
    R = 6370.0
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

# TODO: should plot line segments that also connect templates together in the event that a match is not a new event
def plot_matches(fam_df, fig, legend_colors, num_min=3):
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
        yval = [row['longitude']] * inum
        color = legend_colors[inum-1]
        print(yval)
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

# use plotly
def plot(df, num_min, title):
    '''
    Plot all templates in df with datetime on X-axis, location along strike on Y-axis.
    (NOTE: currently using longitude as proxy for distance along strike)
    :param df:
    :param title: title of plot
    :return:
    '''
    '''
    Use Plotly discrete color scale: https://plotly.com/python/discrete-color/
    There are 10 values (indexed from 0 to 9) available.
    Values are offset to fit in range 0 to 9, but I don't think they're scaled.
    I am coloring the plot by number of matches for each template event, so the max should be 10.
    '''
    if True:
        # num_det are string values - want to get discrete colors
        max_det = pd.to_numeric(df['num_det']).max()
        legend_vals = [str(i) for i in range(1,max_det+1)]
    else:
        # num_det are int values - but then you get a continuous color spectrum
        max_det = df['num_det'].max()
        legend_vals = [str(i) for i in range(1,max_det+1)]
    print('legend: {}'.format(legend_vals))
    legend_colors = px.colors.qualitative.Plotly    # a list of RGB values
    #fig = px.scatter(df, x='templ_dt', y='longitude', color='num_det', title=title)    # unsorted legend
    # plot the template events as filled circles
    fig = px.scatter(df, x='templ_dt', y='longitude', color='num_det', title=title,
                     category_orders={'num_det': legend_vals}, hover_data=['latitude','longitude'])
    fig.update_traces(marker={'size':12})
    # plot the families
    plot_matches(df, fig, legend_colors, num_min=num_min)
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
        print('templ_dt={}, det_dts: {}'.format(tdt, det_dts))
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

def main(quality, is_new, num_min=3):
    new_ev = 1 in is_new
    old_ev = 0 in is_new
    title = 'Quality={}, New Matches={}, Template Matches={}, Min Family={}'.format(quality, new_ev, old_ev, num_min)
    templates = EqTemplates()
    matches = MatchImages(IMAGE_DIR, templates)
    matches.filter(quality=quality, is_new=is_new)
    family_df = getFamilyEvents(templates, matches)
    family_df['n_sta'] = None
    print(family_df)
    det_dfs = getDetectionFiles()
    attachStationCounts(family_df, det_dfs)
    #print(family_df)
    plot(family_df, num_min, title)

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