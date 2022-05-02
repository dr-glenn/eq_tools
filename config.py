# Common parameters for my programs
import datetime as dt

LOCAL=True     # True if running on my desktop, False on "real" computers
# Some of the programs should never be run LOCAL, since GB of mseed files are required,
# so not all parameters here have valid values for a local run.

if LOCAL:
    PROG_DIR = 'E:\Glenn Nelson\science\eq_gaps'
    FILES_DIR = PROG_DIR
    QUAKEML_DIR = FILES_DIR+'/quakeml/'
    # DETECTION are for match_filter or Tribe.detect outputs
else:
    PROG_DIR = '/home/gnelson/work/eq_gaps'
    FILES_DIR = '/proj/shumagin/gnelson'
    QUAKEML_DIR = '/proj/shumagin/quakeml/'
    # DETECTION are for match_filter or Tribe.detect outputs
MSEED_DIR = FILES_DIR+'/mseed/'  # all station daily files
DATA_HOME = FILES_DIR+'/data/'   # all station mseed organized by network
TEMPLATE_DIR = FILES_DIR+'/templates/'
TEMPLATE_PLOTS = FILES_DIR+'/plots/templates/'
DETECTION_DIR = FILES_DIR+'/detections/'
DETECTION_PLOTS = FILES_DIR+'/plots/detections/'
STATION_FILE = FILES_DIR+'/station_merge.csv'
EV_REGION_FILE = FILES_DIR+'/ev_selected_region.csv'
MATCH_RECORD_FILE = FILES_DIR+'/match_records.csv'
EV_CATALOG = 'cat_out.xml'  # output from quakeml_filter.py
EV_CSV = 'ev_selected.csv'  # output from quakeml_filter.py
STATIONS_FILE = 'station_loc.csv'   # not used?

DT_MATCH_VALUE = dt.timedelta(seconds=30) # time in seconds to decide if times match

########################################
### Parameters for EqCorrscan
########################################
SAMP_RATE = 40  # resample HHZ channels from 100. BHZ channels are already 40.
FILTER = (4.0, 10.0)
TEMPL_LEN = 4.0 # seconds
TEMPL_PREPICK = 0.5
PROC_LEN = 86400    # seconds (1 day)
TEMPL_SNR = 3.0     # minimum signal/noise
TEMPL_PLOT_SIZE = (12,10)   # inches
########################################

########################################
### Polygon parallel to trench for event selection
TEMPLATE_POLY = [ (-160.5,52.5), (-162.5,54.8), (-152.0,58.5), (-149.5,56.6), (-160.5,52.5) ]
