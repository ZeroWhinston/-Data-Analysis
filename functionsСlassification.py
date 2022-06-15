import pandas as pd
from sklearn.cluster import KMeans

def KmeansFunc(df):
    column = ['EI', 'CoefSR', 'CoopTime', 'DocAvgNorm', 'DocDelayAvgNorm', 'DocRatioOfDelayed']
    all_pred = KMeans(n_clusters = 11, init='k-means++', n_init = 20, max_iter = 1000, algorithm='full').fit_predict(df[column])
    df['CategoryOfPartner'] = pd.DataFrame(all_pred, columns=['CategoryOfPartner'])
    return df

def OneHotFunc(df):
    df_new = df['CategoryOfPartner']
    df_new = pd.get_dummies(df_new, columns=['CategoryOfPartner'], prefix='Risk')
    return df_new

