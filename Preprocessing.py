import pandas as pd
import glob
import os
from datetime import timedelta
from numpy import nan


def read_datafolder(path):
    """ function to read a whole folder"""
    all_files = glob.glob(os.path.join(path, "*.CSV"))
    df_from_each_file = (pd.read_csv(f, encoding="latin-1") for f in all_files)
    full_df = pd.concat(df_from_each_file, ignore_index=True).drop_duplicates()
    full_df = full_df[full_df.SYSDATE != 0]
    return full_df


def preprocess_branch_data(datafolder_path, save_path):
    # read datafolder for branch
    df = read_datafolder(datafolder_path).sort_values('JOURID')

    # create datetime from SYSDATE and SYSTIME
    df['SYSTIME'] = df['SYSTIME'].apply(lambda x: str(x).zfill(6))
    df['datetime'] = pd.to_datetime(df['SYSDATE'].astype(str) + " " + df['SYSTIME'].astype(str), format='%Y%m%d %H:%M:%S')
    # set timedelta of 4 hours to add sales after midnight to day before
    df.datetime = df.datetime - timedelta(hours=4)
    # format datetime column to datetime
    df = df.groupby([pd.Grouper(freq='D', key='datetime')]).NETTO.sum()
    df = df.loc[:'2017-12-31']

    # drop last day of february in leap-year
    if df.index.contains(pd.to_datetime('2012-02-29')):
        df = df.drop(pd.to_datetime('2012-02-29'))
    if df.index.contains(pd.to_datetime('2016-02-29')):
        df = df.drop(pd.to_datetime('2016-02-29'))

    # handle negative values (outlier)
    df.loc[df.values < 0] = nan
    # replace nan values with previous day's value (padding)
    df = df.fillna(method='pad')
    # handle positive outliers
    df.loc[df.values > (3 * df.mean())] = nan
    # replace nan values with previous day's value (padding)
    df = df.fillna(method='pad')
    # save df as pickle
    df.to_pickle(save_path)


# Branch 1-3
preprocess_branch_data("*/*/CompanyB/Filiale_1/*/", "data/preprocessed/branch1.pkl")
preprocess_branch_data("*/*/CompanyB/Filiale_2/*/", "data/preprocessed/branch2.pkl")
preprocess_branch_data("*/*/CompanyB/Filiale_4/*/", "data/preprocessed/branch3.pkl")
# Branch 4-6
preprocess_branch_data("*/*/CompanyA/Filiale_4/", "data/preprocessed/branch4.pkl")
preprocess_branch_data("*/*/CompanyA/Filiale_3/", "data/preprocessed/branch5.pkl")
preprocess_branch_data("*/*/CompanyA/Filiale_6/", "data/preprocessed/branch6.pkl")
# add more branches

print("Data has been preprocessed and saved in /data/preprocessed")
