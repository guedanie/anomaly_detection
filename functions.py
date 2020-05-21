from __future__ import division
import itertools
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import math
from sklearn import metrics
from random import randint
from matplotlib import style
import seaborn as sns

# ------------------- #
#       Acquire       # 
# ------------------- #

def wrangle_access_logs():
    # acquire
    colnames=['date', "time", 'destination', 'unknown_1',
              'unknown_2', "ip"]
    df_orig = pd.read_csv("curriculum-access.txt",          
                     engine='python',
                     header=None,
                     index_col=False,
                     names=colnames,
                     sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])',
                     na_values='"-"',
                         )
    
    # join the date and time
    df_orig["time_stamp"] = pd.to_datetime(df_orig.date + " " + df_orig.time)
    
    # Drop columns since it is all in the time_stamp
    df_orig.drop(columns=["time", "date"], inplace = True)
    
    # Sort values by date
    df = df_orig.sort_values(by="time_stamp")
    
    # Change value types
    df["unknown_2"] = df["unknown_2"].astype(float)
    
    # Rename the unknown columns
    df = df.rename(columns={"unknown_1": "user_id", "unknown_2": "cohort"})
    
    # Set the index to date
    df = df.set_index("time_stamp")
    
    # Create new columns detailing if student is web_dev or data_science
    web_dev = df[df.destination == "java-ii"].groupby("user_id").user_id.sum().index

    df['is_wd'] = df['user_id'].apply(lambda x: 1 if x in web_dev else 0)

    ds = df[df.destination == "1-fundamentals/1.1-intro-to-data-science"].groupby("user_id").user_id.sum().index

    df['is_ds'] = df['user_id'].apply(lambda x: 1 if x in ds else 0)
    
    return df

# ------------------- #
#      Evaluation     #
# ------------------- #

# to calculate percent - b
def create_pct_b(df, target_variable, span, bound=3):
    span = span
    ema = df.ewm(span=span, adjust=False).mean()
    stdev = df.ewm(span=span, adjust=False).std()
    
    previous_val = pd.DataFrame({'previous_val': df[target_variable]})
    prev_day_df = stdev.join(ema, how='left', lsuffix="_x")

    prev_day_df = prev_day_df.join(previous_val, how='left')

    prev_day_df.fillna(0, inplace = True)

    my_index = df.index[1:]

    prev_day_df = prev_day_df[:-1].reset_index().set_index(my_index)

    prev_day_df.drop(columns=["time_stamp", target_variable + "_x", target_variable], inplace=True)

    target_val = pd.DataFrame({'target_val': df[target_variable]})

    df = target_val.join(prev_day_df, how='left')

    df.fillna(0, inplace = True)

    df["ema"] = ema
    df["stdev"] = stdev

    # compute the upper and lower band
    df['ub'] = df['ema'] + bound*df['stdev']
    df['lb'] = df['ema'] - bound*df['stdev']

    # compute percent b
    df['pct_b'] = (df['target_val'] - df['lb']) / (df['ub'] - df['lb'])
    
    return df

# --------------------- #
#       Analysis        #
# --------------------- #

def run_general_analysis(logs, target_variable,  date = 2):
    '''
    Goes through the entire dataset, and adds a percent_b to every item.
    Returns the df, and prints any unexpected values. Looks at overall results.
    '''
    df = logs.groupby("time_stamp")[[target_variable]].count().resample("D").sum()
    date = date
    active = True
    while active:
        if date < df.shape[0]:
            date += 1
            df_1 = df.iloc[:date]
            df_1 = create_pct_b(df, target_variable, 2)
        else:
            active = False

    years = ["2018", "2019", "2020"]
    for col in years:
        if (df_1.pct_b > 1).sum() == 0:
            print(f"No anomaly activity detected in {col}")
        elif (df_1.pct_b > 1).sum() > 0:
            print()
            print(f"Suscicious report(s) in {col}:")
            print("-------")
            print(f"Number of high reports: {df[col][df[col].pct_b >= 1].shape[0]}")
            print(df[col][df[col].pct_b >= 1][["target_val", "pct_b"]])
            print()
            print(f"Number of low reports: {df[col][df[col].pct_b < 0].shape[0]}")
            print(df[col][df[col].pct_b < 0][["target_val", "pct_b"]])
            print()
            
    return df_1

def run_cohort_analysis(logs, target_variable, cohort=0):
    '''
    Analysis if there is any unexpected findings on a cohort basis
    Returns dataframe with percent_b, and alerts of any findings.
    '''
    df = pd.DataFrame()
    cohorts = logs.cohort.value_counts().index.sort_values()
    cohort = 0
    active = True
    while active:
        if cohort < cohorts.shape[0]:
            test = logs[logs.cohort == cohorts[cohort]].groupby("time_stamp")[[target_variable]].count().resample("D").sum()
            value = create_pct_b(test, target_variable, 2)
            value["cohort"] = cohorts[cohort]
            df = pd.concat([df, value])
            cohort += 1
            if df.pct_b[-1] > 1:
                print(f'There is a high anomaly detected in cohort {cohorts[cohort]} on {df.iloc[-1].name}')
            elif df.pct_b[-1] < 0:
                print(f'There is a low anomaly detected in cohort {cohorts[cohort]} on {df.iloc[-1].name}')
        else:
            active = False

    years = ["2018", "2019", "2020"]
    for col in years:
        if (df.pct_b > 1).sum() == 0:
            print(f"No anomaly activity detected in {col}")
        elif (df.pct_b > 1).sum() > 0:
            print()
            print(f"Suscicious report(s) in {col}:")
            print("-------")
            print(f"Number of high reports: {df[col][df[col].pct_b >= 1].shape[0]}")
            print(df[col][df[col].pct_b >= 1][["cohort","target_val", "pct_b"]])
            print()
            print(f"Number of high reports: {df[col][df[col].pct_b >= 1].shape[0]}")
            print(df[col][df[col].pct_b < 0][["cohort","target_val", "pct_b"]])
            print()

    return df

def run_user_analysis(logs, target_variable, user=0):
    df = pd.DataFrame()
    users = logs.user_id.value_counts().index.sort_values()
    user = 0
    active = True
    while active:
        if user < users.shape[0]:
            test = logs[logs.user_id == users[user]].groupby("time_stamp")[["destination"]].count().resample("D").sum()
            value = create_pct_b(test, "destination", 2)
            value["user_id"] = users[user]
            df = pd.concat([df, value])
            user += 1
            if df.pct_b[-1] > 1:
                print(f'There is a high anomaly detected in user {users[user]} on {df.iloc[-1].name}')
            elif df.pct_b[-1] < 0:
                print(f'There is a low anomaly detected in cohort {users[user]} on {df.iloc[-1].name}')
        else:
            active = False

    years = ["2018", "2019", "2020"]
    for col in years:
        if (df.pct_b > 1).sum() == 0:
            print(f"No anomaly activity detected in {col}")
        elif (df.pct_b > 1).sum() > 0:
            print()
            print(f"Suspicious report(s) in {col}:")
            print("-------")
            print(f"Number of high reports: {df[col][df[col].pct_b >= 1].shape[0]}")
            print(df[col][df[col].pct_b >= 1][["user_id","target_val", "pct_b"]].sort_values(by="pct_b", ascending=False))
            print()
            print(f"Number of low reports: {df[col][df[col].pct_b < 0].shape[0]}")
            print(df[col][df[col].pct_b < 0][["user_id","target_val", "pct_b"]].sort_values(by="pct_b", ascending=False))
            print()
            
    return df