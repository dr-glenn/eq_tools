# experiment with plotly to make a multi-line scatter plot with different symbols
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import math

# use plotly
def plot(df, title=None):
    '''
    Plot all templates in df with datetime on X-axis, location along strike on Y-axis.
    (NOTE: currently using longitude as proxy for distance along strike)
    :param df:
    :param title: title of plot
    :return:
    '''
    # TODO: use itertools.cycle() to change colors? See: https://community.plotly.com/t/how-to-copy-color-from-one-trace-to-another/47277
    fig = px.scatter(df, y=['y1','y2'], symbol='symbols', symbol_map='identity')
    # with changing symbol, this creates a broken line: only same symbols are connected by line
    #fig = px.line(df, y=['y1','y2'], symbol='sym')
    fig.update_traces(marker={'size':12}, mode='lines+markers')
    fig.show()

def plot1(df, title=None):
    '''
    Plot all templates in df with datetime on X-axis, location along strike on Y-axis.
    (NOTE: currently using longitude as proxy for distance along strike)
    :param df:
    :param title: title of plot
    :return:
    '''
    # plots 3 lines and uses the 'symbols' column from df
    print(df)
    fig = px.scatter(df, x=df.index, y=['y1','y2','y3'], symbol='symbols', symbol_map='identity')
    fig['data'][0]['line']['color'] = 'green'
    fig['data'][1]['line']['color'] = 'orange'
    # with changing symbol, this creates a broken line: only same symbols are connected by line
    #fig = px.line(df, y=['y1','y2'], symbol='sym')

    # Now plot only the first point of each line as a different symbol
    row = df.iloc[0]
    d = {'y1': [1,], 'y2':[2,], 'symbols':['diamond',]}
    df1 = pd.DataFrame(data=d, index=[0,])
    print(type(df1))
    print(df1)
    #df1['symbol'] = 'diamond'
    #fig.add_scatter(x=df1.index, y=df1.y1, mode='markers', marker_symbol='diamond', marker_size=10, showlegend=False)
    #fig.add_scatter(x=df1.index, y=df1.y2, mode='markers', marker_symbol='diamond', marker_size=10, showlegend=False)
    marker1 = dict(size=16, symbol='diamond', color='blue')
    marker2 = dict(size=16, symbol='diamond', color='red')
    fig.add_scatter(x=df1.index, y=df1.y1, mode='markers', marker=marker1, showlegend=False)
    fig.add_scatter(x=df1.index, y=df1.y2, mode='markers', marker=marker2, showlegend=False)
    fig.update_traces(marker={'size':12}, mode='lines+markers') # overrides previous size settings
    fig.show()

def plot2(df, title=None):
    '''
    Plot all templates in df with datetime on X-axis, location along strike on Y-axis.
    (NOTE: currently using longitude as proxy for distance along strike)
    :param df:
    :param title: title of plot
    :return:
    '''
    # plots 3 lines and uses the 'symbols' column from df
    print(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df.y1, legendgroup='y1+y3', legendgrouptitle_text='y1+y3',
                             name='y1 line', mode='lines+markers',
                             line=dict(color='red'), marker=dict(color='red',size=12,symbol='diamond-open')))
    fig.add_trace(go.Scatter(x=df.index, y=df.y3, legendgroup='y1+y3', legendgrouptitle_text='y1+y3',
                             name='y3 line', mode='lines+markers', line=dict(color='red'), showlegend=False))
    fig.add_trace(go.Scatter(x=df.index, y=df.y2, legendgroup='y2', legendgrouptitle_text='y2',
                             name='y2 line', mode='lines+markers', showlegend=True,
                             line=dict(color='blue'), marker=dict(color='blue',size=12,symbol='diamond-open')))
    # filled markers for first point on each line
    fig.add_trace(go.Scatter(x=[df.index[0],], y=[df.y1[0],], legendgroup='y1+y3 origin', legendgrouptitle_text='y1+y3',
                             name='y1 marker', mode='markers', showlegend=False,
                             line=dict(color='red'), marker=dict(color='red',size=12,symbol='diamond')))
    fig.add_trace(go.Scatter(x=[df.index[0]], y=[df.y3[0]], legendgroup='y1+y3 origin', legendgrouptitle_text='y1+y3',
                             name='y3 marker', mode='markers', showlegend=False,
                             line=dict(color='red'), marker=dict(color='red',size=12,symbol='diamond')))
    fig.add_trace(go.Scatter(x=[df.index[0]], y=[df.y2[0]], legendgroup='y2 origin', legendgrouptitle_text='y2',
                             name='y2 line', mode='markers', showlegend=False,
                             line=dict(color='blue'), marker=dict(color='blue',size=12,symbol='diamond')))

    # Now plot only the first point of each line as a different symbol
    d = {'y1': [1,], 'y2':[2,], 'symbols':['diamond',]}
    df1 = pd.DataFrame(data=d, index=[0,])
    '''
    print(type(df1))
    print(df1)
    marker1 = dict(size=16, symbol='diamond', color='blue')
    marker2 = dict(size=16, symbol='diamond', color='red')
    fig.add_scatter(x=df1.index, y=df1.y1, mode='markers', marker=marker1, showlegend=False)
    fig.add_scatter(x=df1.index, y=df1.y2, mode='markers', marker=marker2, showlegend=False)
    fig.update_traces(marker={'size':12}, mode='lines+markers') # overrides previous size settings
    '''
    fig.show()

def make_df():
    # XY values. First column is X, other columns are Y
    #symbols = ['circle-open','circle-open','diamond-open','diamond-open','diamond-open']
    symbols = ['diamond-open','diamond-open','diamond-open','diamond-open','diamond-open']
    d = {'y1':[1,2,3,4,5], 'y2':[2,3,4,5,6], 'y3':[3,4,5,6,7], 'symbols':symbols}
    df = pd.DataFrame(data=d, index=[0, 1, 2, 3, 4])
    return df

def main():
    #title = 'Quality={}, IS_New={}, Min Match={}'.format(quality, is_new, num_min)
    df = make_df()
    plot2(df)

if __name__ == '__main__':
    main()