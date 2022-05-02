import pandas as pd

from config import STATION_FILE


class Stations:
    # all station locations for plotting on maps
    def __init__(self, station_file=STATION_FILE):
        self.df = pd.read_csv(station_file)