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
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.dates import drange
from matplotlib import cm   # colormap
import pandas as pd
import math
from templates import EqTemplates
from matches import MatchImages
from config import MATCH_RECORD_FILE,STATION_FILE,EV_REGION_FILE,DETECTION_PLOTS

IMAGE_DIR = DETECTION_PLOTS

from functools import wraps
from time import time
from my_util import dt_match

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

@timing
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
    templ_list = []     # it will be a list of lists; each element is a single template event + all new detections
    for index,row in detect_df.iterrows():
        templ = templates.find(row['templ_dt'])     # lookup template event info for the current row
        if not templ.empty:
            templ_row = templ.iloc[0].tolist()      # convert event info to a list
            templ_row.append(row['det_dt'])         # append list of all matched detections
            templ_list.append(templ_row)
        else:
            print('SKIP: no match for {}'.format(row['templ_dt']))

    df_new = pd.DataFrame(data=templ_list, columns=['time','longitude','latitude','depth','mag','region','template','templ_file','templ_dt','det_dt'])
    return df_new

def plot(df, title):
    # TODO: should use distance along strike
    # plot using longitude as proxy for distance along strike
    maxlon = df['longitude'].max()
    minlon = df['longitude'].min()
    # TODO: cannot do min/max of det_dt, because it is a list of times.
    '''
    mintime = (df[['templ_dt','det_dt']].min())
    maxtime = (df[['templ_dt','det_dt']].max())
    '''
    # TODO: maybe should round down/up the times to month boundaries
    mintime = df['templ_dt'].min()
    maxtime = df['templ_dt'].max()
    print('longitude limits = {}, {}'.format(minlon, maxlon))
    print('time limits = {}, {}'.format(mintime, maxtime))

    # X axis is time, Y axis is distance
    fig, axs = plt.subplots(figsize=(12,10))
    delta = dt.timedelta(days = 7)
    dates = drange(mintime, maxtime, delta)
    #dates = df['templ_dt'].values.tolist() # nope - converts to long int
    dates = df['templ_dt']
    print(dates[:5])
    dists = df['longitude'].values.tolist()
    print(dists[:5])
    num_match = df['det_dt'].apply(lambda det: len(det) if len(det) < 8 else 8)
    zmax = 8
    my_cm = cm.get_cmap(name='rainbow', lut=zmax)    # sequence is violet-blue-green-orange-red
    print(num_match[:5])
    # NOTE! plot_date cannot assign different colors to each point.
    #axs.plot_date(dates,dists, c=num_match, cmap=plt.get_cmap('Greens'))
    scatter = axs.scatter(dates, dists, c=num_match, cmap=my_cm)
    axs.legend(*scatter.legend_elements(), title='Matches', loc='lower right')
    plt.title(title)
    plt.show()

def main(quality, is_new):
    title = 'Quality={}, IS_New={}'.format(quality, is_new)
    templates = EqTemplates()
    matches = MatchImages(IMAGE_DIR, templates)
    matches.filter(quality=quality, is_new=is_new)
    family_df = getFamilyEvents(templates, matches)
    print(family_df)
    plot(family_df, title)

if __name__ == '__main__':
    quality = [3,4]
    is_new  = [1,]
    main(quality, is_new)