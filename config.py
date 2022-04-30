# Common parameters for my programs
LOCAL=True     # True if running on my desktop, False on "real" computers
# Some of the programs should never be run LOCAL, since GB of mseed files are required,
# so not all parameters here have valid values for a local run.

if LOCAL:
    MSEED_DIR = './mseed/'  # all station daily files
    DATA_HOME = './data/'   # all station mseed organized by network
    TEMPLATE_DIR = './templates/'
    TEMPLATE_PLOTS = './plots/templates/'
    QUAKEML_DIR = './quakeml/'
    # DETECTION are for match_filter or Tribe.detect outputs
    DETECTION_DIR = './detections/'
    DETECTION_PLOTS = './plots/detections/'
else:
    MSEED_DIR = '/proj/shumagin/gnelson/mseed/'  # all station daily files
    DATA_HOME = '/proj/shumagin/gnelson/data/'   # all station mseed organized by network
    TEMPLATE_DIR = '/proj/shumagin/gnelson/templates/'
    TEMPLATE_PLOTS = '/proj/shumagin/gnelson/plots/templates/'
    QUAKEML_DIR = '/proj/shumagin/quakeml/'
    # DETECTION are for match_filter or Tribe.detect outputs
    DETECTION_DIR = '/proj/shumagin/gnelson/detections/'
    DETECTION_PLOTS = '/proj/shumagin/gnelson/plots/detections/'
EV_CATALOG = 'cat_out.xml'  # output from quakeml_filter.py
EV_CSV = 'ev_selected.csv'  # output from quakeml_filter.py
STATIONS_FILE = 'station_loc.csv'
SAMP_RATE = 40  # resample HHZ channels from 100. BHZ channels are already 40.
FILTER = (4.0, 10.0)
TEMPL_LEN = 4.0 # seconds
TEMPL_PREPICK = 0.5
PROC_LEN = 86400    # seconds (1 day)
TEMPL_SNR = 3.0     # minimum signal/noise
TEMPL_PLOT_SIZE = (12,10)   # inches
# Polygon parallel to trench for event selection
TEMPLATE_POLY = [ (-160.5,52.5), (-162.5,54.8), (-152.0,58.5), (-149.5,56.6), (-160.5,52.5) ]
