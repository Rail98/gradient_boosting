import pandas as pd
from load_weekend import load_weekend
from env import params
import warnings
warnings.filterwarnings("ignore")


def create_features(df, target_name='fact',
                    lag_max=0, rolling_step=0, 
                    lag_predict=24,
                    file_path=None, db_user=None, db_password=None, table_name='weekend'
                    ):
    
    #выходные дни
    df_ = load_weekend(df,
                       db_user=db_user, db_password=db_password, table_name='weekend')
    
    #календарные дни
    #df_['year'] = df_.index.year
    #df_['quarter'] = df_.index.quarter
    df_['month'] = df_.index.month
    #df_['week'] = df_.index.isocalendar().week
    df_['weekday'] = df_.index.weekday
    df_['hour'] = df_.index.hour
    
    #lag_max
    for lag in range(1, lag_max+1):
        df_[f'lag_{lag}'] = df_[target_name].shift(lag) 
    
    #rolling_step
    if rolling_step > 1:
        df_['rolling_mean'] = df_[target_name].shift().rolling(rolling_step).mean()
        df_['rolling_std'] = df_[target_name].shift().rolling(rolling_step).std()
           
    #lag_predict
    if lag_predict is not None:
        column_predict = []
        for lag in range(0, lag_predict):
            df_[f'{target_name}_{lag}'] = df_[target_name].shift(-lag)
            column_predict.append(f'{target_name}_{lag}')
    
    df_ = df_.drop(columns=[target_name])
    df_ = df_.dropna()
    
    if lag_predict is not None:
        df_features = df_.drop(columns=column_predict)
        df_target = df_[column_predict]
        return df_features, df_target
    else:
        df_features = df_
        return df_features


if __name__ == "__main__":
    df_features, df_target = create_features.create_features(df, target_name='fact',
                                                             lag_max=0, rolling_step=0, lag_predict=24, #предсказываем на день вперед
                                                             file_path=None, db_user=None, db_password=None, table_name='weekend')
    print(df) 




