from pytrends.request import TrendReq
import pandas as pd
from datetime import datetime

def get_gt_historical_test(kw_list):
    # enter list of keywords as arguemnts to fetch google trends data for
    # from a year ago up until now

    pytrends = TrendReq(hl='en-US', tz=0) # 0 = UCT, 60 = CET

    # specifies current time as end date of results
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    now_d = int(now[:2])
    now_m = int(now[3:5])
    now_y = int(now[6:10])
    now_h = int(now[11:13])

    # '''list of keywords to get data, in the future, should we have a big list in
    # a csv we fetch them from? empty list is needed because API works with lists,
    # and we want to feed it one coin at a time'''

    coin_list = []
    data = []

    for kw in kw_list:
        coin_list = [kw]
        data.append(pytrends.get_historical_interest(
            coin_list, year_start=now_y-1, month_start=now_m, day_start=24,
            hour_start=00, year_end=now_y, month_end=now_m, day_end=n24,
            hour_end=00, cat=0, sleep=0))

    # create first df from first coin in list
    df = data[0].iloc[:,0]

    # merges all coins into one df
    if len(kw_list) > 1:
        for i in range(1,len(kw_list)):
            df = pd.merge(left=df, right=data[i].iloc[:,0], how='outer', on='date')

    # non-existing values are replaced by 0.25,
    # (avg of 0 & 0.5, the range of values marked as <1 on GT)
    df = df.fillna(0.25)

    #how to get in other function?
    last_timestamp = df.index[-1]

    return df


def get_gt_historical(kw_list):
    # enter list of keywords as arguemnts to fetch google trends data for
    # from a year ago up until now

    pytrends = TrendReq(hl='en-US', tz=0) # 0 = UCT, 60 = CET

    # specifies current time as end date of results
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    now_d = int(now[:2])
    now_m = int(now[3:5])
    now_y = int(now[6:10])
    now_h = int(now[11:13])

    # '''list of keywords to get data, in the future, should we have a big list in
    # a csv we fetch them from? empty list is needed because API works with lists,
    # and we want to feed it one coin at a time'''

    coin_list = []
    data = []

    for kw in kw_list:
        coin_list = [kw]
        data.append(pytrends.get_historical_interest(
            coin_list, year_start=now_y-1, month_start=now_m, day_start=now_d,
            hour_start=now_h, year_end=now_y, month_end=now_m, day_end=now_d,
            hour_end=now_h, cat=0, sleep=0))

    # create first df from first coin in list
    df = data[0].iloc[:,0]

    # merges all coins into one df
    if len(kw_list) > 1:
        for i in range(1,len(kw_list)):
            df = pd.merge(left=df, right=data[i].iloc[:,0], how='outer', on='date')

    # non-existing values are replaced by 0.25,
    # (avg of 0 & 0.5, the range of values marked as <1 on GT)
    df = df.fillna(0.25)

    #how to get in other function?
    last_timestamp = df.index[-1]

    return df


def get_gt_current(kw_list):

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
    for kw in kw_list:
        scaler = df[kw].iloc[-1] / df_new[kw].iloc[0]
        df_new[kw] = scaler * df_new[kw]

    df = pd.concat([df, df_new.iloc[1:,:]], axis=0)

    return df
