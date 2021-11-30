from numpy.lib.function_base import diff
import pandas as pd
import datetime as datetime
from pycoingecko import CoinGeckoAPI
from time import sleep
from pytrends.request import TrendReq
from crypto_prediction.utils import date2utc_ts, gecko_make_df
import pytz
from crypto_prediction.params import COIN_TRANSLATION_TABLE

def _one_coin_financial_history(gecko_id, vs_currency, start_dt, end_dt):
    """
    gets the hourly values of a single coin, dont call alone, needs tests and calculations
    from coin_history()

    input:
        gecko-id, vs_currency, start_dt (<class 'datetime.datetime'>), end_dt (<class 'datetime.datetime'>)

    output:
        dataframe - index is datetime
    """
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    cg = CoinGeckoAPI()
    gecko_raw = cg.get_coin_market_chart_range_by_id(id=gecko_id,
                                                vs_currency=vs_currency,
                                                from_timestamp=start_ts,
                                                to_timestamp=end_ts
                                                )
    return gecko_raw

def coin_history(tickerlist, start, end = 'now'):
    """
    returns historical coin data

    input:
        tickerlist      - list of ticker names, will be translated to coingecko
        start           - 2021-12-30T13:12:00Z (utc) OR integer as HOURS (aka cycles) from end
        end             - 2021-12-30T13:12:00Z (utc) OR default (now)

        throws an error if start - end is < 0 (wrong time) or > 90 days from the time of the query(!), because:
                            CoinGecko Data granularity is automatic (cannot be adjusted)
                            1 day from query time = 5 minute interval data (has to be fitted to 1 hour)
                            1 - 90 days from query time = hourly data
                            above 90 days from query time = daily data (00:00 UTC) (has to be discarded)
    output:
        returns dict of dataframes
        {ticker: dataframe,
         ticker: dataframe, ..}
    """
    # get the time and date in order
    now_dt = datetime.datetime.now(datetime.timezone.utc)

    if end == 'now':
        end_dt = now_dt # default end is now
    else:
        end_dt = datetime.datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)

    if isinstance(start, int):
        # start is integer, start hours (not days) before end-date
        if start > 1:
            start = start - 1
        start_dt =  end_dt - datetime.timedelta(days=start/24)
    else:
        # start is a normal date
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)

    # tests to check the < 0 and > 90 day rule, start to NOW, not end
    diff_days = (now_dt - start_dt).days + 1
    assert diff_days >=  0, str(diff_days) + ' days for days of data grabbing at coingecko too short (0..90) allowed'
    assert diff_days <= 90, str(diff_days) + ' days for days of data grabbing at coingecko too long (0..90) allowed'

    # loop over the coins in the list, calls one_coin_financial_history() for a single coin
    coins_dict = {}
    for ticker in tickerlist:
        assert ticker in COIN_TRANSLATION_TABLE, 'tickername ' + ticker + ' not in COIN_TRANSLATION_TABLE, gecko ID not in reach'
        gecko_id = COIN_TRANSLATION_TABLE[ticker]['id_coingecko']
        raw_coin_data = _one_coin_financial_history(gecko_id,
                                               'eur',
                                               start_dt,
                                               end_dt
                                               )

        df_coin_data = gecko_make_df(raw_coin_data)

        assert isinstance(df_coin_data, pd.DataFrame), '_one_coin_financial_history() did not return a dataframe (but it should)'

        coins_dict[ticker] = df_coin_data

        # coingecko has 50 calls / minute max, so if we have to many coins, sleep a while inbetween
        if len(tickerlist) > 20: sleep(1)
        if len(tickerlist) > 40: sleep(2)

    return coins_dict




def googletrend_history(tickerlist, start, end = 'now'):
    """
    gets the hourly trend-data
    input:
        namelist        - list of coin names, they will be translated to their (atm: single) searchterm
        start           - 2021-12-30T13:12:00Z (utc) OR integer as HOURS (aka cycles) from end
        end             - 2021-12-30T13:12:00Z (utc) OR default (now)
    output:
        dataframe       - with every name in the namelist as columns and the date as index
    """

    # get the time and date in order
    now_dt = datetime.datetime.now(datetime.timezone.utc)

    if end == 'now':
        end_dt = now_dt # default end is now
    else:
        end_dt = datetime.datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)

    if isinstance(start, int):
        # start is integer, start hours before end-date
        if start > 1:
            start = start - 1
        start_dt =  end_dt - datetime.timedelta(days=start/24)
    else:
        # start is a normal date
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)

    # transform UTC-timestring to datetime-object
    #start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
    #end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")

    # setup the engine
    pytrends = TrendReq(hl='en-US', tz=0)  # 0 = UCT, 60 = CET

    data = []
    for ticker in tickerlist:
        # translate from ticker to trendsearch
        assert ticker in COIN_TRANSLATION_TABLE, 'tickername ">>>' + ticker + '"<<< not in COIN_TRANSLATION_TABLE, gecko ID not in reach'
        searchname = COIN_TRANSLATION_TABLE[ticker]['trend']

        data.append(
            pytrends.get_historical_interest([searchname],
                                             year_start=start_dt.year,
                                             month_start=start_dt.month,
                                             day_start=start_dt.day,
                                             hour_start=start_dt.hour,
                                             year_end=end_dt.year,
                                             month_end=end_dt.month,
                                             day_end=end_dt.day,
                                             hour_end=end_dt.hour,
                                             cat=0,
                                             sleep=5))

    df = data[0].iloc[:, 0]

    # merges all coins into one df
    if len(tickerlist) > 1:
        for i in range(1, len(tickerlist)):
            df = pd.merge(left=df,
                          right=data[i].iloc[:, 0],
                          how='outer',
                          on='date')

    df = pd.DataFrame(df)
    # make renaming dict
    renaming_dict = {}  # {'old_col1':'new_col1', 'old_col2':'new_col2', ...}
    for ticker in tickerlist:
       renaming_dict[COIN_TRANSLATION_TABLE[ticker]['trend']] = ticker
    # use renaming dict
    df.rename(columns = renaming_dict, inplace = True)

    # non-existing values are replaced by 0.25,
    # (avg of 0 & 0.5, the range of values marked as <1 on GT)
    df = df.fillna(0.25)

    # set the index as to type datetime64[ns] to be ready for grouping
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')

    #how to get in other function?
    #last_timestamp = df.index[-1]

    return df


def prediction_ready_df(tickerlist, model_history_size = 2):
    """
    gets the last model_history_size dates from trends and prices and fits it in a neat df

    output:
        list of dataframes      - [price_(tickername)] [trend_(tickername)], index datetime
    """

    # get all coins prices by ticker from now minus model_history_size
    coins_dict = coin_history(tickerlist, model_history_size)

    # get the trend data for the coins
    trends_df = googletrend_history(tickerlist, model_history_size)

    # loop over each coin
    predict_me = []
    for ticker in tickerlist:
        # drop columns from prices and make df
        df = coins_dict[ticker].drop(columns=['timestamp','market_caps','total_volumes'])
        # merge part of the trends into price-df
        df = pd.merge(df, trends_df[ticker], how='left', left_index=True, right_index=True)
        # fill nans with -1 in the trend-column if some left from merging
        df[ticker].fillna(-1,inplace=True)
        # rename columns
        df.rename(columns = {'price':'price_'+ticker, ticker:'trend_'+ticker}, inplace = True)
        predict_me.append(df)

    return predict_me


def google_trends_current(kw_list):

    # enter list of keywords as arguemnts to fetch google trends data for
    # from hour of last value in historical data until now

    pytrends = TrendReq(hl='en-US', tz=0) # 0 = UCT, 60 = CET

    # specifies current time as end date of results
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    now_d = int(now[:2])
    now_m = int(now[3:5])
    now_y = int(now[6:10])
    now_h = int(now[11:13])

    # '''list of keywords to get data, in the future, should we have a big list in
    # a csv we fetch them from? empty list is needed because API works with lists,
    # and we want to feed it one coin at a time
    # TO DO: how to get timestamp'''

    coin_list = []
    data = []

    for kw in kw_list:
        coin_list = [kw]
        data.append(pytrends.get_historical_interest(
            coin_list, year_start=last_timestamp.year, month_start=last_timestamp.month, day_start=last_timestamp.day,
            hour_start=last_timestamp.hour, year_end=now_y, month_end=now_m, day_end=now_d,
            hour_end=now_h, cat=0, sleep=0))

    # turns data into df

    df_new = data[0].iloc[:,0]
    if len(kw_list) > 1:
        for i in range(1,len(kw_list)):
            df_new = pd.merge(left=df_new, right=data[i].iloc[:,0], how='outer', on='date')

    # non-existing values are replaced by 0.25 (avg of 0 & 0.5, the range of values marked as <1 on GT)
    df_new = df_new.fillna(0.25)

    # differences in scale between old and new data
    # TO DO: need to add exception if coin not in list
    # TO DO: need to communicate with old data to get df
    # df here refers to the old df with the hitorical data
    for kw in kw_list:
        scaler = df[kw].iloc[-1] / df_new[kw].iloc[0]
        df_new[kw] = scaler * df_new[kw]

    df = pd.concat([df, df_new.iloc[1:,:]], axis=0)

    return df

if __name__ == "__main__":
    # ------------------- just for quick csv-saves -------------------
    #_hourly_coin_static_csv('samoyedcoin', '2021-08-28T00:00:00Z', '2021-11-26T00:00:00Z', write=True)
    #df = googletrend_history(['dogecoin', 'samoyedcoin'], '2021-11-20T00:00:00Z', '2021-11-26T00:00:00Z')
    #print(df)
    #print(df)
    # ----------------------------------------------------------------

    # quick tests:

    #df = googletrend_history(['doge', 'samo'], '2021-11-23T12:00:00Z', '2021-11-24T00:00:00Z')
    #df = googletrend_history(['doge', 'samo'], 2)
    #print(df)

    #print(coin_history(['doge'], '2021-10-28T08:00:00Z'))
    #print(prediction_ready_df(['doge','shib','elon','samo','hoge','mona','dogedash','erc20','ban','cummies','doggy','smi','doe','pepecash','wow','dinu','yummy','shibx','lowb','grlc'], 2160))
    #print(prediction_ready_df(['doge'], 10))
    pass
