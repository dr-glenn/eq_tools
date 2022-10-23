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
import plotly.graph_objects as go
import pandas as pd
import haversine as hv
from templates import EqTemplates
from matches import MatchImages, getFamilyEvents
from families import Families
import config as cfg
from config import MATCH_RECORD_FILE,STATION_FILE,EV_REGION_FILE,DETECTION_PLOTS

IMAGE_DIR = DETECTION_PLOTS
TEMPLATE_POLY = cfg.TEMPLATE_POLY

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
    '''
    Between ll0 and ll1 we have divided the study region into nzone. Assumes the region is rectangular.
    The returned values are distance in km from ll0.
    :param ll0: (longitude,latitude) at one end of study region
    :param ll1: (longitude,latitude) at other end of study region
    :param nzone: number of equally spaced zones within region
    :return: list of zone boundaries, inclusive of the endpoints.
    '''
    # TODO: should optioanlly be able to return (long,lat) coordinates of zone boundaries.
    a12,c12,d12,bearing12 = hv.haversine(ll0, ll1)
    print('region length = {} km'.format(d12))
    dzone = d12 / nzone
    zones = [dzone * float(i) for i in range(0,nzone+1)]
    return zones

# TODO: should plot line segments that also connect templates together in the event that a match is not a new event
def plot_matches(fam_df, fig, legend_colors, num_min=3, yaxis='longitude'):
    '''
    Using family_df, add lines to template location plot where each line shows new event detections
    with datetime on X-axis and distance along strike on Y-axis.
    Since we don't have locations for these match detections, we use same location as template.
    Each detection time is marked with an open symbol.
    :param fam_df: the DF with families of events
    :param fig: plotly figure, this function adds to it
    :param legend_colors: list of discrete colors in the legend of the original plot
    :param num_min: minimum number of new detected events to plot (so that the figure isn't too cluttered)
    :param yaxis: column name in fam_df for y-axis. 'longitude' or 'dist' are supported.
    :return:
    '''
    n_color = len(legend_colors)
    df = fam_df[['templ_dt',yaxis,'det_dt','num_det','n_sta']].copy()
    #df['inum_det'] = pd.to_numeric(df['num_det'])
    max_det = df['num_det'].max()
    # We plot all templates with same number of matches with the same color.
    # Thus we select all templates in df that have same 'inum_det' and plot them as one group.
    for n_det in range(num_min, max_det+1):
        new_df = df[df['num_det'] == n_det]
        # Construct a single array for all template matches with same n_det.
        # Put breaks in the array between events, using NaN.
        xarr = []
        yarr = []
        hovertexts = []
        for index,row in new_df.iterrows():
            inum = row['num_det']   # TODO: should be same as n_det?
            nc = inum if inum <= n_color else n_color
            # add each match time to xarr
            xarr.extend(row['det_dt'])
            xarr.extend([row['templ_dt']])
            xarr.extend([float('nan')]) # break the line in order to separate from the next template matches
            yval = [row[yaxis]] * (inum+1)  # all family events have same y-value as the template
            yarr.extend(yval)
            yarr.extend([float('nan')]) # break the line
        for i in range(len(xarr)):
            if xarr[i] != float('nan'):
                # TODO: the points that represent templ_dt should have lat-long info too
                hovertexts.append('{}'.format(xarr[i])) # hover: datetime
                #hovertexts.append('{}<br>{}'.format(xarr[i], row['n_sta'])) # hover: datetime + num stations
            else:
                hovertexts.append('NaN')
        print('len(xarr)={}, len(hovertexts)={}'.format(len(xarr), len(hovertexts)))
        lg_name = 'N={}'.format(inum)   # was inum
        legendgroup = 'templ_match'
        color = legend_colors[nc-1]
        # plot all the templates that have inum matches
        fig.add_trace(go.Scatter(x=xarr, y=yarr,
                                 legendgroup=legendgroup, legendgrouptitle_text='Template Matches',
                                 name=lg_name, mode='lines+markers',
                                 hoverinfo='text', hovertext=hovertexts,
                                 line=dict(color=color), marker=dict(color=color,size=12,symbol='diamond-open')))

def plot_zones(fig, xval, zones):
    '''
    Plot markers along Y-axis to designate the zones within our study region
    :param fig:
    :param xval: location for annotating the zone boundaries (along time axis)
    :param zones: list of zone boundaries inclusive of start and end region bounds
    :return:
    '''
    # Since zones[] includes endpoints, there is one less zone inside
    x = [xval] * len(zones)
    fig.add_scatter(x=x[1:-1], y=zones[1:-1],
                    mode='markers', marker_color='black', marker_size=16, marker_symbol='triangle-right',
                    showlegend=False)
    ytext = [(zones[i]+zones[i+1])/2.0 for i in range(0,len(zones)-1)]
    text = ['Zone {}'.format(i) for i in range(0,len(zones)-1)]
    print('Zone arrays: {}, {}'.format(len(ytext), len(text)))
    fig.add_scatter(x=x, y=ytext, text=text, mode='text', showlegend=False)

# uses plotly
def plot(df, num_min, title, yaxis='longitude', file_out=None):
    '''
    Plot all templates in df with datetime on X-axis, location along strike on Y-axis.
    This function only plots template events and color codes them for number of matches found.
    (NOTE: currently using longitude as proxy for distance along strike)
    :param df: DataFrame of families
    :param num_min: minimum family size to display as timelines
    :param title: title of plot
    :param yaxis: 'longitude' or 'dist'. If 'dist', then mark zone boundaries on yaxis
    :param file_out: if specified, write HTML file, else display locally
    :return:
    '''
    print('plot {}'.format(yaxis))
    fig = go.Figure()
    fig.update_layout(title=dict(text=title, font=dict(color='blue',size=24)))
    #fig.update_yaxes(title=dict(text=yaxis.upper()))
    fig.update_yaxes(title_text=yaxis.upper())  # alternate way to specify a title, plotly allows '_'
    '''
    Use Plotly discrete color scale: https://plotly.com/python/discrete-color/
    There are 10 values (indexed from 0 to 9) available.
    Values are offset to fit in range 0 to 9, but I don't think they're scaled.
    I am coloring the plot by number of matches for each template event, so the max should be 10.
    '''
    # num_det are int values - but then you get a continuous color spectrum - so convert to str
    max_det = df['num_det'].max()
    #df['num_det_str'] = df['num_det'].apply(lambda x: str(int(x)))  # convert int to str
    legend_vals = [str(i) for i in range(1,max_det+1)]
    print('legend values: {}'.format(legend_vals))
    legend_colors = px.colors.qualitative.Plotly    # a list of RGB values
    n_color = len(legend_colors)

    # plot the families (matches) that are affiliated with a template
    # TODO: maybe plot_matches first so that the family roots appear on top layer of figure
    plot_matches(df, fig, legend_colors, num_min=num_min, yaxis=yaxis)

    for n_det in range(1,max_det+1):
        dfn = df[df['num_det']==n_det]
        if dfn.empty:
            print('no events with {} matches'.format(n_det))
            continue
        nc = n_det if n_det <= n_color else n_color
        lg_name = 'N={}'.format(n_det)

        color = legend_colors[nc-1]
        legendgroup = 'templ_ev'
        hovertexts = []
        for idx,row in dfn.iterrows():
            if row['n_sta']:
                n_sta = str(row['n_sta'])
            else:
                n_sta = 'NA'
            hovertexts.append('{}<br>Lon: {}<br>Lat: {}<br>N Sta: {}'.
                              format(row['templ_dt'], row['longitude'], row['latitude'], n_sta))
        # plots all template events as filled symbol, color designates the number of matches that each template found
        fig.add_trace(go.Scatter(x=dfn.templ_dt, y=dfn[yaxis],
                                 legendgroup=legendgroup, legendgrouptitle_text='Template Events',
                                 name=lg_name, mode='markers', showlegend=True,
                                 line=dict(color=color), marker=dict(color=color,size=12,symbol='diamond'),
                                 hovertext=hovertexts, hoverinfo='text'))

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

    # toggleitem means that individual entries in a hierarchical legend can be turned on/off.
    # default is that the entire hierarchy is turned on/off.
    # For example, earthquake magnitudes 1,2,3,4,5,6,7,8. With toggleitem, you can turn off display
    # of all events of magnitude 4. Without toggleitem, clicking on any one of the magnitude values in the legend
    # will turn off ALL magnitudes.
    fig.update_layout(legend=dict(groupclick='toggleitem'))
    if not file_out:
        fig.show()  # live display in your local browser. Will not work over VPN.
    else:
        # Write HTML file that references plotly.js with URL (does not write full copy of JS)
        # Copy the HTML file to your local computer or email.
        fig.write_html(file_out+'.html', include_plotlyjs='cdn')

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
        # NOTE: all events in family have same number of stations
        # TODO: should not need to do this next loop
        for det_dt in det_dts:
            match_df = df[df['detect_time'] == det_dt]
            if not match_df.empty:
                n = df[df['detect_time'] == det_dt].iloc[0].no_chans
                #print('n = {}'.format(n))
                n_sta.append(n)
            else:
                # TODO: I don't know how this can happen
                print('match_df is empty: det_dt = {}'.format(det_dt))
        #row['n_sta'] = n_sta
        if len(n_sta) > 0:
            family_df.loc[index, 'n_sta'] = n_sta[0]
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

def main(quality, is_new, yaxis, num_min=3, file_out=None):
    new_ev = 1 in is_new
    old_ev = 0 in is_new
    title = 'Quality={}, New Matches={}, Template Matches={}, Min Family={}'.format(quality, new_ev, old_ev, num_min)
    templates = EqTemplates()
    matches = MatchImages(IMAGE_DIR, templates)
    matches.filter(quality=quality, is_new=is_new)
    my_df = matches.getSelectedDF()
    print('main: {} rows in filtered DF'.format(my_df.shape[0]))
    # TODO: look at this family DF and compare with new better method
    match_fams = matches.family()
    match_fams.to_csv('match_fams.csv')

    fams = Families(my_df)
    # TODO: compare with 'match_fams.csv'
    fams.export('families.csv', root_only=False)
    fams.show()    # creates prt file with synopsis of all families

    family_df = getFamilyEvents(templates, fams.getFamilies())
    family_df['n_sta']  = None
    family_df['dist']   = None
    family_df['offset'] = None
    locationInBox(family_df)
    print(family_df)
    print('dist: {} to {}'.format(family_df['dist'].min(), family_df['dist'].max()))
    print('offset: {} to {}'.format(family_df['offset'].min(), family_df['offset'].max()))
    if False:   # TODO
        det_dfs = getDetectionFiles()
        attachStationCounts(family_df, det_dfs)
    #print(family_df)
    plot(family_df, num_min, title, yaxis=yaxis, file_out=file_out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display time plot of template families',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('-h','--help')
    parser.add_argument('-f', '--file', dest='file', type=str, default=None, help='Output HTML file, otherwise display locally')
    parser.add_argument('-n', '--num_detect', dest='num_min', type=int, default=1, help='minimum number of detections in family')
    parser.add_argument('-i', '--include_template_matches', dest='include_template_matches', default=False, action='store_true',
                        help='include matches between templates, else only show new detections')
    parser.add_argument('-q', '--quality', dest='quality', type=str, default='3,4', help='match quality levels to display (1,2,3,4)')
    parser.add_argument('-y', '--yaxis', dest='yaxis', type=str, default='dist', help='distance along strike, longitude, or depth')
    args = parser.parse_args()
    if args.include_template_matches:
        is_new = [0,1]
    else:
        is_new = [0, ]  # only show templates that match templates, not show new detections
    quality = [int(x) for x in args.quality.split(',')]
    main(quality, is_new, args.yaxis, num_min=args.num_min, file_out=args.file)