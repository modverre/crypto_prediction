from numpy.lib.function_base import diff
import pandas as pd
import datetime as datetime
from pycoingecko import CoinGeckoAPI
from time import sleep
from pytrends.request import TrendReq
from crypto_prediction.utils import gecko_make_df, twitter_make_df
import pytz
from crypto_prediction.params import COIN_TRANSLATION_TABLE
import os
from dotenv import load_dotenv, find_dotenv
import requests

load_dotenv(find_dotenv()) # automatic find

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
        tweak = False
        if start > 1:
            start = start - 1
        else:
            tweak = True
        start_dt =  end_dt - datetime.timedelta(days=start/24)
    else:
        # start is a normal date
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)

    # tests to check the < 0 and > 90 day rule, start to NOW, not end
    diff_days = (now_dt - start_dt).days
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
        # important, otherwise it returns 2 rows if only 1 is called (for every other case its fine)
        if tweak:
            df_coin_data = df_coin_data.iloc[1: , :]

        coins_dict[ticker] = df_coin_data

        # coingecko has 50 calls / minute max, so if we have to many coins, sleep a while inbetween
        if len(tickerlist) > 20: sleep(1)
        if len(tickerlist) > 40: sleep(2)

    return coins_dict


def twitter_history(tickerlist, start, end = 'now'):
    """
    returns historical twitter data

    input:
        tickerlist      - list of ticker names, will be translated to coingecko
        start           - 2021-12-30T13:12:00Z (utc) OR integer as HOURS (aka cycles) from end
        end             - 2021-12-30T13:12:00Z (utc) OR default (now)

        throws an error if start - end is < 0 (wrong time) or > 90 days from the time of the query to keep it simple
    output:
        returns dict of dataframes
        {ticker: dataframe,
         ticker: dataframe, ..}
    """
    # twitter likes questions only 10seconds older than now, lets make it 30seconds
    now_dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=30)
    if end == 'now':
        end_dt = now_dt # default end is now
    else:
        end_dt = datetime.datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)

    if isinstance(start, int):
        # start is integer, start hours (not days) before end-date
        start_dt =  end_dt - datetime.timedelta(days=start/24)
    else:
        # start is a normal date
        start_dt = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)

    # tests to check the < 0 and > 90 day rule, start to NOW, not end
    diff_days = (now_dt - start_dt).days + 1
    assert diff_days >=  0, str(diff_days) + ' days for days of data grabbing at twitter too short (0..90) allowed'
    assert diff_days <= 90, str(diff_days) + ' days for days of data grabbing at twitter too long (0..90) allowed'

    # get token
    TOKEN = os.getenv('TOKEN')
    # create header
    headers = {"Authorization": "Bearer {}".format(TOKEN)}

    return_dict = {}
    for ticker in tickerlist:
        keyword = COIN_TRANSLATION_TABLE[ticker]['twitter']
        start_date = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        end_date = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        results_list = []
        # first call
        url,  query_params = _twitter_create_url(keyword, start_date,end_date)
        json_response = _twitter_connect_to_endpoint(url, headers, query_params)
        data = json_response["data"]
        results_list.append(data)
        # if paginated data
        while True:
            next_token = json_response['meta'].get('next_token', None)
            if next_token is None:
                break
            else:
                query_params["next_token"] = next_token
                #time.sleep(2)
                json_response = _twitter_connect_to_endpoint(url, headers, params=query_params, next_token = next_token)
                data = json_response["data"]
                results_list.append(data)
        # its a paginated list anyway (and its in the wrong direction, so we have to reverse order first)
        results_list = results_list[::-1]
        result = []
        for _ in results_list:
            result = result + _
        return_dict[ticker] = twitter_make_df(result)
        sleep(1) # for good measure
    return return_dict

def _twitter_create_url(keyword, start_date, end_date, granularity = 'hour'):
    search_url = "https://api.twitter.com/2/tweets/counts/all"
    query_params = {'query':keyword ,
                    'start_time': start_date,
                    'end_time': end_date,
                    'granularity': granularity,
                    'next_token': {}}
    return search_url, query_params

def _twitter_connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    #print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


def prediction_ready_df(tickerlist, model_history_size = 2):
    """
    gets the last model_history_size dates from prices and twitter and fits it in a neat df

    output:
        list of dataframes      - [price_(tickername)] [trend_(tickername)], index datetime
    """

    # get all coins prices by ticker from now minus model_history_size
    coins_dict = coin_history(tickerlist, model_history_size)

    # twitter
    twitter_dict = twitter_history(tickerlist, model_history_size)

    # debug
    # return coins_dict, twitter_dict

    # loop over each coin
    predict_me = []
    for ticker in tickerlist:
        # drop columns from prices and make df
        df = coins_dict[ticker].drop(columns=['timestamp','market_caps','total_volumes'])
        # merge part of the trends into price-df
        df = pd.merge(df, twitter_dict[ticker], how='left', left_index=True, right_index=True)
        # fill nans with -1 in the trend-column if some left from merging
        # df[ticker].fillna(-1,inplace=True)
        # rename columns
        df.rename(columns = {'price':'price_'+ticker, 'tweet_count':'tweets_'+ticker}, inplace = True)
        predict_me.append(df)

    return predict_me



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
    #df = twitter_history(['ban', 'doge'], 10)
    #print(df)

    #test = prediction_ready_df2(['doge','ban'], 100)
    #print(test)
    #ttest = twitter_history(['ban', 'doge'], 10)

    #print(ttest['ban'])

    pass
