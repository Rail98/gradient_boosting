import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#самостоятельно
def data_split_handler(df, 
                       split_date=None, 
                       target_hour=None, 
                       show_info=True):
                       
    df_train = df.copy()
    df_test = df.copy()
    
    if split_date is not None:
        df_train = df_train[:split_date - pd.to_timedelta(1, unit='h')] #[:-1]исключаем послденюю запись, так как она попадает на тестовую выборку
        df_test = df_test[split_date:]

    #оставляем записи, в промежутке которого будет выполняться прогноз
    if target_hour is not None:
        df_train = df_train[df_train.index.hour == target_hour]
        df_test = df_test[df_test.index.hour == target_hour]
        
    #показать размер выборок
    if show_info == True:
        print(f'Размер обучающей выборки: {df_train.shape} ({round(len(df_train)/(len(df_train) + len(df_test))*100, 2)}% от общей выборки)')
        print(f'Размер тестовой выборки: {df_test.shape} ({round(len(df_test)/(len(df_train) + len(df_test))*100, 2)}% от общей выборки)')
    return df_train, df_test


if __name__ == "__main__":
    start_date_test = '2024-01-01 00:00:00'
    target_hour = 0
    df_features_train, df_features_test = data_split_handler(df_features, 
                                                             start_date_test, 
                                                             target_hour=target_hour, 
                                                             show_info=True)
    df_target_train, df_target_test = data_split_handler(df_target, 
                                                         start_date_test, 
                                                         hour=target_hour, 
                                                         show_info=False)
    print(df_features_train)




