# quick fix - merge 2 CSV files that have data for each station
# The unique index is column 1 and 2: network and station IDs.
# 1. station_loc.csv contains latitude, longitude and elevation
# 2. station_dates.csv contains date0 and date1: the range of the data that we have.
#    If a station has no data then the dates are "NA"

# I thought I could easily do this in Excel, but it's not obvious that it will be easy.
# Use Pandas because it can create an index from multiple columns.

import pandas as pd
from obspy import UTCDateTime

fname1 = 'station_loc.csv'
fname2 = 'station_dates.csv'
merge_name = 'station_merge.csv'

# TODO: need to specify columns that contain dates
df1 = pd.read_csv(fname1, header=0, index_col=None)
df2 = pd.read_csv(fname2, header=0, index_col=None)
#df2.describe()
# columns for output
columns = ['network', 'station', 'longitude', 'latitude', 'elevation',
        'has_data', 'date0', 'date1', 'CHANE', 'CHANN', 'CHANZ', 'deploy_date0', 'deploy_date1']
#df_out = pd.DataFrame(columns=columns)

data_out = []

# use itertuples to preserve dtype
for row in df1.itertuples():
    #print(row.station, row.latitude, row.longitude)
    # find matching station in df2
    row2 = df2[df2['station'].str.match(row.station)]
    if not row2.empty:
        data2 = row2.iloc[0]
        if row2['date0'].isnull().values.any() or row2['date1'].isnull().values.any():
            has_data = 'N'
        else:
            has_data = 'Y'
        data = [row.network, row.station, row.longitude, row.latitude, row.elevation,
                has_data, data2.date0, data2.date1, data2.CHANE, data2.CHANN, data2.CHANZ,
                row.deploy_date0, row.deploy_date1]
        #print(row2.iloc[0])
        #print(data)
    else:
        data = [row.network, row.station, row.longitude, row.latitude, row.elevation,
                'N']
    data_out.append(data)   # a list of lists

df_out = pd.DataFrame(data_out, columns=columns)
df_out.to_csv(merge_name)
   
print(fname1)
print(df1.columns)
print(df1.dtypes)
print(fname2)
print(df2.columns)
print(df2.dtypes)
