from utils import *
from data import googletrend_history, coinlist_financial_history#, one_coin_financial_history
import datetime as datetime

COIN_TRANSLATION_TABLE = {
    'doge' : {
        'trend': 'dogecoin',
        'coingecko': 'dogecoin',
        'display': 'Doge'
    },
    'shiba-inu' : {
        'trend': 'shiba-inu coin',
        'coingecko': 'shiba-inu',
        'display': 'Shiba-Inu'
    },
    'samoyed' : {
        'trend': 'samoyedcoin',
        'coingecko': 'samoyecdoin',
        'display': 'Samoyed'
    }
}

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
    coin_raw = coinlist_financial_history([name_gecko], start_date_hack, end_date, interval = '1d')

    # get and translate the coin-df into prediction-ready df
    df = coin_raw[name_gecko].tail(MODEL_HISTORY_SIZE)
    df.rename(columns = {'price':'high'}, inplace = True)
    df.drop(['timestamp', 'market_caps', 'total_volumes'], axis=1, inplace=True)

    # get and translate the trends-df into prediction-ready df
    name_trend = COIN_TRANSLATION_TABLE[coin_name]['trend']
    df_trend = googletrend_history([name_trend], start_date, end_date, interval = '1d').tail(MODEL_HISTORY_SIZE) # otherwise gets 3 instead of 2

    # putting together
    df['Google_trends'] = df_trend[name_trend]

    return df

if __name__ == "__main__":
    prediction_ready_df('doge')
