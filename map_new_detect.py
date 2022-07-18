# (run with Anaconda env match_conda_2)
# Map new events that match templates.
# Since we only have match_filter results, there is no location.
# So for each template, count the number of matches and plot on map with size or color for number of matches.

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm   # colormap
from mpl_toolkits.basemap import Basemap

from templates import EqTemplates
from matches import MatchImages, getFamilyEvents
from config import FILES_DIR,DETECTION_PLOTS,MATCH_RECORD_FILE,STATION_FILE,EV_REGION_FILE
from my_util import dt_match

ONLY_NEW = False

def mapSetup(axes, bbox):
    '''
    Setup a map object, it's an instance of Basemap
    :param axes: returned from Figure
    :param bbox: tuple(tuple(lower-left long,lat), tuple(upper-right long, lat))
    :return: basemap
    '''
    # use low resolution coastlines.
    map = Basemap(projection='cyl',llcrnrlat=bbox[0][1], urcrnrlat=bbox[1][1],
                  llcrnrlon=bbox[0][0], urcrnrlon=bbox[1][0], resolution='l', ax=axes)
    # draw coastlines, country boundaries, fill continents.
    map.drawcoastlines(linewidth=0.25)
    map.drawcountries(linewidth=0.25)
    map.fillcontinents(color='coral',lake_color='aqua')
    # draw the edge of the map projection region (the projection limb)
    #map.drawmapboundary(fill_color='aqua')
    map.drawmapboundary(fill_color='lightgrey')
    # draw lat/lon grid lines every 1 degrees.
    map.drawmeridians(np.arange(190,212,1), labels=[0,0,0,1])
    map.drawparallels(np.arange(52,59,1), labels=[1,1,0,0])
    return map

def makeMapPlot(map_axes, stations, events):
    '''
    Map of the region. Displayed in a QtWidget
    :param ax1: matplotlib Axes object from a Figure
    :param stations: DataFrame of seismic stations
    :param events: DataFrame of earthquakes
    :return:
    '''
    # plot map
    # llcrnrlat=52, urcrnrlat=59, llcrnrlon=-162, urcrnrlon=-150
    bound_box = ((-164.0, 52.0),(-149.0, 59.0))
    map = mapSetup(map_axes, bound_box)
    # plot stations
    # plot events
    return map

# read template events
templates = EqTemplates()
# read matches, filter and generate family DataFrame
matches = MatchImages(DETECTION_PLOTS, templates)
quality = [3,4]
if ONLY_NEW: is_new = [1,]
else: is_new = [0,1]
matches.filter(quality=quality, is_new=is_new)
df_new = getFamilyEvents(templates, matches)

# plot on map
fig,ax = plt.subplots(figsize=(10,9))
m = makeMapPlot(ax, None, None)
# just plot all events

lons = df_new['longitude'].tolist()
lats = df_new['latitude'].tolist()
x,y = m(lons, lats)
z = df_new['num_det'].tolist()
#print(z)
zmin = min(z)
zmax = max(z)
print('zmax = {}'.format(zmax))
# see https://matplotlib.org/stable/tutorials/colors/colormaps.html for various colormaps
#my_cm = cm.get_cmap('viridis', zmax)    # yellow is difficult to see and represents highest number
#my_cm = cm.get_cmap('autumn', zmax)     # colors not distinct enough
#my_cm = cm.get_cmap('brg', zmax)   # difficult to see green against blue ocean
#my_cm = cm.get_cmap('rainbow', zmax)    # sequence is violet-blue-green-orange-red
my_cm = cm.get_cmap('jet', zmax)    # sequence is violet-blue-green-orange-red, but more contrast than rainbow
labels = [str(i+1) for i in range(zmax)]
#m.scatter(x, y, 5, marker='o', color='k')
scatter = m.scatter(x, y, s=30, marker='o', c=z, cmap=my_cm)
ax.legend(*scatter.legend_elements(), title='New Events', loc='lower right')
if ONLY_NEW:
    plt.title('Templates with High Quality Matches (only new)')
    outfile = 'template_matches_new.png'
else:
    plt.title('Templates with High Quality Matches (include other templates)')
    outfile = 'template_matches_all.png'

#plt.show()
plt.savefig(FILES_DIR+outfile)
