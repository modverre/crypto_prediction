import pandas as pd
import datetime as datetime
from pycoingecko import CoinGeckoAPI
from time import sleep
from pytrends.request import TrendReq
from utils import *


# if you try to fetch less than them: dont
# used to be 80 but then get trouble with the short dates for the prediction
# also its just a hack
MIN_COIN_HISTORY_DATAPOINTS = 80


def one_coin_financial_history(gecko_id, vs_currency, start_date, end_date, interval='1d'):
    """
    input:
        data for a single coin

    output:
        dataframe per interval, maybe it does not have the complete length of start_date to end_date
        because there is not enough data on coingecko
    """
    try:
        # translate date (should be utc) to timestamp
        from_timestamp = date2utc_ts(start_date)
        to_timestamp = date2utc_ts(end_date)
    except:
        return 'error in date transformation for a single coin'

    try:
        cg = CoinGeckoAPI()
        gecko_raw = cg.get_coin_market_chart_range_by_id(id=gecko_id,
                                                vs_currency=vs_currency,
                                                from_timestamp=from_timestamp,
                                                to_timestamp=to_timestamp
                                                )
        return gecko_make_df(gecko_raw)
    except:
        return 'couldnt get a dataframe from the coingecko-call, debug me'



def coinlist_financial_history(gecko_ids, start_date, end_date, interval='1d'):
    """
    input:
        gecko_ids       - list of ids as found in coingecko
        start_date      - 2021-12-30T13:12:00Z (utc format)
        end_date        - 2021-12-30T13:12:00Z (utc format)
        interval='1d'   - unused for coingecko:
                            Data granularity is automatic (cannot be adjusted)
                            1 day from query time = 5 minute interval data
                            1 - 90 days from query time = hourly data
                            above 90 days from query time = daily data (00:00 UTC)

    output:
        returns dict of dataframes
        {gecko_id: dataframe,
         gecko_id: dataframe, ..}
    """

    coins_dict = {}
    for gecko_id in gecko_ids:
        coin_data = one_coin_financial_history(gecko_id,
                                               'eur',
                                               start_date,
                                               end_date
                                               )
        if not isinstance(coin_data, pd.DataFrame):
            return f'error: "{coin_data}"'
        else:
            if coin_data.shape[0] < MIN_COIN_HISTORY_DATAPOINTS:
                # leave it out of the dict
                print(f'coin {gecko_id} only has {coin_data.shape[0]} datapoints instead of {MIN_COIN_HISTORY_DATAPOINTS} and will be excluded.')
            else:
                # put it in the dict
                coins_dict[gecko_id] = coin_data

        # coingecko has 50 calls / minute max, so if we have to many coins sleep a while inbetween
        if len(gecko_ids) > 10: sleep(1)
        if len(gecko_ids) > 20: sleep(2)

    return coins_dict


def googletrend_history(namelist, start_date, end_date, interval = '1d'):
    """
    gehts the trend-data, daily or hourly
    input:
        namelist        - list of coin names, they will be translated to their (atm: single) searchterm
        start_date      - 2021-12-30T13:12:00Z (utc format)
        end_date        - 2021-12-30T13:12:00Z (utc format)
        interval        - granularity, 1d or 1h
    output:
        dataframe       - with every name in the namelist as columns and the date as index
    """
    # transform UTC-timestring to datetime-object
    start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")

    # setup the engine
    pytrends = TrendReq(hl='en-US', tz=0)  # 0 = UCT, 60 = CET

    data = []
    for name in namelist:
        data.append(
            pytrends.get_historical_interest([name],
                                             year_start=start_dt.year,
                                             month_start=start_dt.month,
                                             day_start=start_dt.day,
                                             hour_start=start_dt.hour,
                                             year_end=end_dt.year,
                                             month_end=end_dt.month,
                                             day_end=end_dt.day,
                                             hour_end=end_dt.hour,
                                             cat=0,
                                             sleep=0))

    df = data[0].iloc[:, 0]

    # merges all coins into one df
    if len(namelist) > 1:
        for i in range(1, len(namelist)):
            df = pd.merge(left=df,
                          right=data[i].iloc[:, 0],
                          how='outer',
                          on='date')

    # non-existing values are replaced by 0.25,
    # (avg of 0 & 0.5, the range of values marked as <1 on GT)
    df = df.fillna(0.25)

    # set the index as to type datetime64[ns] to be ready for grouping
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')

    #how to get in other function?
    #last_timestamp = df.index[-1]

    if interval == '1d':
        daily_df = df.groupby(pd.Grouper(freq='d')).mean()
        daily_df = pd.DataFrame(daily_df)
        return daily_df
    return df

if __name__ == "__main__":
    pass
    # quick tests:
    #df = coinlist_financial_history(['samoyedcoin', 'dogecoin'], '2020-11-24T00:00:00Z', '2021-11-24T00:00:00Z')
    #print(df)
    #print(len(df))

    df = googletrend_history(['dogecoin'], '2021-11-23T00:00:00Z', '2021-11-24T00:00:00Z')
    print(df)
