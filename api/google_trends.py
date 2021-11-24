# outputs a dataframe with historical google trends data

from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime

pytrends = TrendReq(hl='en-US', tz=60) # 60 = CET

# specifies current time as end date of results

now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
now_d = int(now[:2])
now_m = int(now[3:5])
now_y = int(now[6:10])
now_h = int(now[11:13])

# list of keywords to get data, in the future, should we have a big list in a
# csv we fetch them from?
# empty list is needed because API works with lists,
# and we want to feed it one coin at a time

kw_list = ['Dogecoin','Samoyedcoin',
 'Hoge Finance','Dogelon Mars','Shiba Inu']
coin_list = []
data = []

def get_google_data():
    for kw in kw_list:
        coin_list = [kw]
        data.append(pytrends.get_historical_interest(
            coin_list, year_start=now_y-1, month_start=now_m, day_start=now_d,
            hour_start=now_h, year_end=now_y, month_end=now_m, day_end=now_d,
            hour_end=now_h, cat=0, sleep=0))

    # create first df from first coin in list
    df = data[0].iloc[:,0]

    # merges all coins into one df
    for i in range(1,len(kw_list)):
        df = pd.merge(left=df, right=data[i].iloc[:,0], how='outer', on='date')

    # non-existing values are replaced by 0
    df = df.fillna(0)

    return df
