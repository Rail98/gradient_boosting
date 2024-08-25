import pandas as pd
from load_weekend import load_weekend
from env import params
import warnings
warnings.filterwarnings("ignore")

#поиск и исключение аномалий
def detect_anomaly(df, target_name='fact',
                   file_path=None, db_user=None, db_password=None,
                   group_list=['weekend', 'month', 'hour'], 
                   k_value=1.5, agg='median', 
                   df_total=None,
                   show_info=True, 
                   activate=True,
                   this_year=False):  
    
    #подготовка данных
    df_ = load_weekend(df,
                       file_path=None,
                       db_user=db_user, db_password=db_password, table_name='weekend')

    df_['datetime'] = df_.index
    df_['year'] = df_.index.year
    df_['quarter'] = df_.index.quarter
    df_['month'] = df_.index.month
    df_['week'] = df_.index.isocalendar().week
    df_['weekday'] = df_.index.weekday
    df_['hour'] = df_.index.hour

    #чтобы определить, аномальное ли значение или нет, найдем медиану на конкретную месяц-день-час
    df_q25 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.25).rename(columns={target_name: 
                                                                                                   f'{target_name}_q25'})
    df_q75 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.75).rename(columns={target_name: 
                                                                                                   f'{target_name}_q75'})
    df_q2 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.02).rename(columns={target_name: 
                                                                                                  f'{target_name}_q2'})
    df_q98 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.98).rename(columns={target_name: 
                                                                                                   f'{target_name}_q98'})
    df_q1 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.01).rename(columns={target_name: 
                                                                                                  f'{target_name}_q1'})
    df_q99 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.99).rename(columns={target_name: 
                                                                                                   f'{target_name}_q99'})
    df_count = df_.groupby(group_list, as_index=False)[target_name].count().rename(columns={target_name: 
                                                                                            f'{target_name}_count'})
    df_mean = df_.groupby(group_list, as_index=False)[target_name].mean().rename(columns={target_name: 
                                                                                          f'{target_name}_mean'})
    df_median = df_.groupby(group_list, as_index=False)[target_name].median().rename(columns={target_name: 
                                                                                              f'{target_name}_median'})
    df_total_ = df_count.merge(df_mean, how='outer').merge(df_median, how='outer')
    df_total_ = df_total_.merge(df_q25, how='outer').merge(df_q75, how='outer')
    df_total_ = df_total_.merge(df_q2, how='outer').merge(df_q98, how='outer')
    df_total_ = df_total_.merge(df_q1, how='outer').merge(df_q99, how='outer')
    
    if df_total is not None:
        df_ = df_.merge(df_total, on=group_list, how='left')
        df_.index = df_['datetime']
    else:
        df_ = df_.merge(df_total_, on=group_list, how='left')
        df_.index = df_['datetime']

    #описательная информация по поиску аномалий
    if show_info == True:
        print(df_[f'{target_name}_count'].value_counts().sort_index()) #кол-во отобранных записей по группировке
        k = 1.5
        anomaly_count = len(df_[(df_[target_name] > df_[f'{target_name}_q75'] + k*(df_[f'{target_name}_q75'] - df_[f'{target_name}_q25'])) \
                                | (df_[target_name] < df_[f'{target_name}_q25'] - k*(df_[f'{target_name}_q75'] - df_[f'{target_name}_q25']))
                                ])
        print(f"Количество выбросов по диаграмму размаха при k=1.5: {anomaly_count} ({round(anomaly_count/len(df_)*100, 2)}%)")
        
        k = 1.2
        anomaly_count = len(df_[(df_[target_name] > df_[f'{target_name}_q75'] + k*(df_[f'{target_name}_q75'] - df_[f'{target_name}_q25'])) \
                                | (df_[target_name] < df_[f'{target_name}_q25'] - k*(df_[f'{target_name}_q75'] - df_[f'{target_name}_q25']))
                                ])
        print(f"Количество выбросов по диаграмму размаха при k=1.2: {anomaly_count} ({round(anomaly_count/len(df_)*100, 2)}%)")
        
        anomaly_count = len(df_[(df_[target_name] > df_[f'{target_name}_q99']) \
                                | (df_[target_name] < df_[f'{target_name}_q1'])
                                ])
        print(f"Количество выбросов от 1% до 99% квантиля: {anomaly_count} ({round(anomaly_count/len(df_)*100, 2)}%)")
        
        anomaly_count = len(df_[(df_[target_name] > df_[f'{target_name}_q98']) \
                                | (df_[target_name] < df_[f'{target_name}_q2'])
                                ])
        print(f"Количество выбросов от 2% до 98% квантиля: {anomaly_count} ({round(anomaly_count/len(df_)*100, 2)}%)")

    #ощищаем датасет от аномалий
    if activate == True:
        if this_year == True:
            def clean_anomaly(row, k):
                if (row['year'] != pd.to_datetime('today').year) \
                    and (row[target_name] < row[f'{target_name}_q25'] - k*(row[f'{target_name}_q75'] - row[f'{target_name}_q25']) \
                         or row[target_name] > row[f'{target_name}_q75'] + k*(row[f'{target_name}_q75'] - row[f'{target_name}_q25'])
                         ):
                    return round(row[f'{target_name}_{agg}'])
                else:
                    return row[target_name]
        else:
            def clean_anomaly(row, k):
                if (row[target_name] < row[f'{target_name}_q25'] - k*(row[f'{target_name}_q75'] - row[f'{target_name}_q25']) \
                    or row[target_name] > row[f'{target_name}_q75'] + k*(row[f'{target_name}_q75'] - row[f'{target_name}_q25'])
                    ):
                    return round(row[f'{target_name}_{agg}'])
                else:
                    return row[target_name]
        df_[target_name] = df_.apply(clean_anomaly, k=k_value, axis=1)
        
    return df_[[target_name]], df_total_


if __name__ == "__main__":
    df = detect_anomaly(df, target_name='fact', 
                        group_list=['weekend', 'month', 'hour'], 
                        k_value=1.5, agg='median', 
                        show_info=True, 
                        activate=True)
    print(df)






