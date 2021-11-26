import pandas as pd
import datetime as datetime
from pycoingecko import CoinGeckoAPI
from time import sleep
from pytrends.request import TrendReq
from crypto_prediction.utils import date2utc_ts, gecko_make_df

COIN_TRANSLATION_TABLE = {
    'doge': {
        'trend': 'dogecoin',
        'coingecko': 'dogecoin',
        'display': 'Doge'
    },
    'shiba-inu': {
        'trend': 'shiba-inu coin',
        'coingecko': 'shiba-inu',
        'display': 'Shiba-Inu'
    },
    'samoyed': {
        'trend': 'samoyedcoin',
        'coingecko': 'samoyecdoin',
        'display': 'Samoyed'
    }
}

# if you try to fetch less than them: dont
# used to be 80 but then get trouble with the short dates for the prediction
# also its just a hack
MIN_COIN_HISTORY_DATAPOINTS = 80


def one_coin_financial_history(gecko_id, vs_currency, start_date, end_date):
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



def coinlist_financial_history(gecko_ids, start_date, end_date):
    """
    input:
        gecko_ids       - list of ids as found in coingecko
        start_date      - 2021-12-30T13:12:00Z (utc format)
        end_date        - 2021-12-30T13:12:00Z (utc format)
                          if its more than 90 days from the time of the query(!) data will be daily
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


def googletrend_history(namelist, start_date, end_date, interval = '1h'):
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

def prediction_ready_df(coin_name):

    # fixed in the model, dont change without changing the model
    MODEL_HISTORY_SIZE = 2
    #build the dates to call the data-getter
    now = datetime.datetime.now(datetime.timezone.utc)
    # dirty hack, if querying within the last 90 days coingecko gives hourly data
    # so we take 90+2 days and then only the last 2 elements for now (time trouble)
    then = now - datetime.timedelta(days=MODEL_HISTORY_SIZE)
    then_hack = now - datetime.timedelta(days=91)

    start_date_hack = then_hack.strftime('%Y-%m-%dT%H:%M:%SZ')
    start_date = then.strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date = now.strftime('%Y-%m-%dT%H:%M:%SZ')

    # get a single coint via the multiple coin-getter
    name_gecko = COIN_TRANSLATION_TABLE[coin_name]['coingecko']
    coin_raw = coinlist_financial_history([name_gecko],
                                          start_date_hack,
                                          end_date)

    # get and translate the coin-df into prediction-ready df
    df = coin_raw[name_gecko]
    df = df.tail(MODEL_HISTORY_SIZE)
    df.rename(columns={'price': 'high'}, inplace=True)
    df.drop(['timestamp', 'market_caps', 'total_volumes'],
            axis=1,
            inplace=True)

    # get and translate the trends-df into prediction-ready df
    name_trend = COIN_TRANSLATION_TABLE[coin_name]['trend']
    df_trend = googletrend_history(
        [name_trend], start_date, end_date, interval='1d').tail(
            MODEL_HISTORY_SIZE)  # otherwise gets 3 instead of 2

    # putting together
    df['Google_trends'] = df_trend[name_trend]

    return df


def hourly_coin_static_csv(gecko_id, start_date, end_date, write=False):
    # if zeit between heute und end date > 90 --> keine hourly mehr
    vs_currency = 'eur'
    from_timestamp = date2utc_ts(start_date)
    to_timestamp = date2utc_ts(end_date)

    cg = CoinGeckoAPI()
    gecko_raw = cg.get_coin_market_chart_range_by_id(id=gecko_id,
                                                vs_currency=vs_currency,
                                                from_timestamp=from_timestamp,
                                                to_timestamp=to_timestamp
                                                )
    df = gecko_make_df(gecko_raw)
    print('df, have a look')
    print(df)

    if write:
        fn = f'{gecko_id}_history_1h_{start_date}---{end_date}.csv'
        print('save csv as ',fn)
        df.to_csv(fn)


def coingecko_ids(part_of_the_name):
    #later or manually
    pass



if __name__ == "__main__":
    # ------------------- just for quick csv-saves -------------------
    #hourly_coin_static_csv('samoyedcoin', '2021-08-28T00:00:00Z', '2021-11-26T00:00:00Z', write=True)
    #df = googletrend_history(['dogecoin', 'samoyedcoin'], '2021-08-28T00:00:00Z', '2021-11-26T00:00:00Z', interval='1h')
    #print(df)
    # ----------------------------------------------------------------

    # quick tests:
    #df = coinlist_financial_history(['samoyedcoin', 'dogecoin'], '2020-11-24T00:00:00Z', '2021-11-24T00:00:00Z')
    #print(df)
    #print(len(df))

    #df = googletrend_history(['dogecoin'], '2021-11-23T00:00:00Z', '2021-11-24T00:00:00Z')
    #print(df)
    pass
