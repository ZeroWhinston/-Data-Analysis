import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def TransformColumn(column, data):
    encoder_df = pd.DataFrame(data, columns=column)
    encoder_df = pd.get_dummies(encoder_df)
    encoder_df = encoder_df.drop(encoder_df.columns[[1]], axis = 1)
    encoder_df.columns = column
    data[column[0]] = encoder_df[column[0]]       
    return data

def TranformColumnPartner(column, data):
    encoder_df = pd.DataFrame(data, columns=column)
    encoder_df = pd.get_dummies(encoder_df)
    return encoder_df

def DropColumn(column, data):
    return data.drop(column, axis = 1)
    
def MinMaxFunc(columns, df_new):
    MinMax = MinMaxScaler()
    df_new[columns + 'Norm'] = MinMax.fit_transform(df_new[columns].values.reshape(-1,1))
    return df_new

def cleanDF(olddf):
    olddf = olddf[:-1]
    olddf = olddf.fillna(0)
    olddf = DropColumn('CalculationObject', olddf)
    olddf = olddf.drop(olddf[((olddf['SalesDebt'] <= 1000) | (olddf['SalesDebt'] >= 30000))].index)
    olddf = olddf.drop(olddf[(olddf['AreaOfActivity'] == 0)].index)
    olddf = TransformColumn(['AreaOfActivity'], olddf)
    olddf = TransformColumn(['EI'], olddf)
    return olddf

def NormalizedDF(olddf):
    df = olddf
    group_column = ['Partner','EI','CoefSR','AreaOfActivity','CoopTime']
    df['DocCount'] = df.groupby(by=group_column, as_index=False)['OverduePayment'].transform(lambda s: s.values.shape[0])
    df['DocAvg'] = df.groupby(by=group_column, as_index=False)['SalesDebt'].transform(lambda s: np.mean(s.values))
    df['DocDelayAvg'] = df.groupby(by=group_column, as_index=False)['DelayDays'].transform(lambda s: np.mean(s.values))
    df['DocDelay'] = 0
    df.loc[(df['DelayDays'] == 0), 'DocDelay'] = 0
    df.loc[(df['DelayDays'] > 0), 'DocDelay'] = 1
    df['DocDelayToday'] = 0
    df.loc[(df['ClientDebt'] == 0), 'DocDelayToday'] = 0
    df.loc[((df['ClientDebt'] > 0) & (df['DelayDays'] > 0)), 'DocDelayToday'] = 1
    df['DocDelayCount'] = df.groupby(by=group_column, as_index=False)['DocDelay'].transform(lambda s: np.sum(s.values))
    df['DocRatioOfDelayed'] = df['DocDelayCount'] / df['DocCount']
    #df = MinMaxFunc("SalesDebt", df)
    df = MinMaxFunc("DocAvg", df)
    df = MinMaxFunc("DocDelayAvg", df)
    df = MinMaxFunc("DocDelayCount", df)
    df = MinMaxFunc("DocCount", df)
    df = DropColumn('OverduePayment', df)
    df = DropColumn('ClientDebt', df)
    df = DropColumn('DelayDays', df)
    #df = DropColumn('DocDelay', df)
    df = DropColumn('DocDelayToday', df)
    #df = DropColumn("SalesDebt", df)
    df = DropColumn("DocAvg", df)
    df = DropColumn("DocDelayAvg", df)
    df = DropColumn("DocDelayCount", df)
    df = DropColumn("DocCount", df)
    df.reset_index(drop=True, inplace=True)
    return df