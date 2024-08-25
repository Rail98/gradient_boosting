
#импорт модулей
import os
import pyodbc
import pandas as pd
from catboost import CatBoostRegressor

from load_weekend import load_weekend
from tools.load_data import load_dataset
from tools.statistic import statistical_analysis
from tools.transform_data import detect_anomaly, create_features, data_split, transform_dataset
from tools.model import create_model, model_eval, model_predict

from env import params


def db_parameters_ods(db_user, db_password):
    server = 'p-ods-db01'
    database = 'master'
    driver = '{SQL Server Native Client 11.0}'
    conn_string = f'SERVER={server};DATABASE={database};DRIVER={driver};UID={db_user};PWD={db_password}'
    return conn_string
    
def db_parameters_glshp(db_user, db_password):
    server = 'p-glshp-db01'
    database = 'ITPerfomanceMetrics'
    driver = '{SQL Server Native Client 11.0}'
    conn_string = f'SERVER={server};DATABASE={database};DRIVER={driver};UID={db_user};PWD={db_password}'
    return conn_string


#оптимальные параметры для каждой модели
#clean_anomaly_flag = True #почистить датасет от аномалий (выбросов)
#boxcox_transform_flag = True #использовать преобразование Бокса-Кокса
#diff1_transform_flag = False #разность первого порядка
#lag_diff1 = 24 #недельная разность
#lag_max = 24 #последние n-часов (лаги)
#lag_predict = 24 #прогноз на n-часов
#target_hour = 0 #время начала прогноза (прогноз планируется выполняться в 00:00:00)
#n_predict = 24 #горизонт прогнозирования - 24 часа
#cat_features = None #['month', 'weekday', 'weekend']

#дополнительные параметры
#target_name = 'fact'
#model_eval_flag = True #отобразить оценки модели
#statistic_flag = False #провести анализ данных
#start_date = '2021-01-01 00:00:00'
#split_date = '2024-01-01 00:00:00'


def unload_data(conn_string,
               query, #запрос возвращает два поля - дата и время операции и id операции
               add_hour=None, #учитывать часовой пояс
               oper_id=True
               ):
    with pyodbc.connect(conn_string) as conn:
        df_ = pd.read_sql(query, conn)
    df_.columns = ['datetime', 'fact']
    df_['datetime'] = pd.to_datetime(df_['datetime'])
    df_ = df_.sort_values('datetime')
    
    if add_hour is not None:  
        df_['datetime'] = df_['datetime'] + pd.Timedelta(hours=add_hour)
        
    df_.index = df_['datetime']
    
    if oper_id:
        df_['fact'] = 1
        
    df_ = df_.resample('1H').sum(numeric_only=True)
        
    return df_[['fact']]


def calculate_model(path, hyperparams, 
                    db_user, db_password, table_name,
                    start_date=None, 
                    end_date=None, split_date_='standart',
                    n_period=1, lag_max=24, lag_predict=24, target_hour=0,
                    n_predict=24, cat_features=None, target_name='fact',
                    anomaly_activate=True,
                    model_eval_flag=True, 
                    save_model=False):
    
    if end_date is None:
        end_date = pd.to_datetime('today').to_period('M').to_timestamp()
        
    if split_date_ == 'standart':
        split_date = end_date - pd.DateOffset(months=n_period)
    elif split_date_ == 'equal':
        split_date = end_date
    else:
        split_date = split_date_
    
    df_train = load_dataset.load_dataset(file_path=None,
                                         db_user=db_user, db_password=db_password, table_name=table_name,
                                         start_date=None, end_date=end_date-pd.to_timedelta(1, unit='h'),
                                         target_name=target_name)
    
    date_start_train = df_train.index.min().strftime('%Y-%m-%d %H:%M:%S')
    date_end_train = split_date.strftime('%Y-%m-%d %H:%M:%S')
    date_start_test = split_date.strftime('%Y-%m-%d %H:%M:%S')
    date_end_test = end_date.strftime('%Y-%m-%d %H:%M:%S')
    date_start_plan = end_date.strftime('%Y-%m-%d %H:%M:%S')
    date_end_plan = (end_date + pd.DateOffset(months=n_period)).strftime('%Y-%m-%d %H:%M:%S')
    #print(f"date_start_train = {date_start_train}, date_end_train = {date_end_train}")
    #print(f"date_start_test = {date_start_test}, date_end_test = {date_end_test}")
    #print(f"date_start_plan = {date_start_plan}, date_end_plan = {date_end_plan}")
    
    #очистка от аномалий
    df_train, df_total = detect_anomaly.detect_anomaly(df_train, target_name=target_name, 
                                                       db_user=db_user, db_password=db_password,
                                                       group_list=['weekend', 'month', 'hour'], 
                                                       k_value=1.5, agg='median', 
                                                       df_total=None,
                                                       show_info=False,
                                                       activate=anomaly_activate,
                                                       this_year=False)
    #df_total.to_excel(os.path.join(path, 'clear_dataset', f"clear_dataset_{table_name}_{end_date.strftime('%d%m%Y')}.xlsx"),
                      #index=False)

    #преобразование Бокса-Кокса
    df_train, const, best_lambda = transform_dataset.boxcox_transform_direct(df_train, target_name=target_name, 
                                                                             const=None, best_lambda=None,
                                                                             plot_kde=False, show_info=False)
    df_boxcox = pd.DataFrame([[end_date, const, best_lambda]], columns=['end_date', 'const', 'best_lambda'])
    #df_boxcox.to_excel(os.path.join(path, 'boxcox_params', f"boxcox_params_{table_name}_{end_date.strftime('%d%m%Y')}.xlsx"),
                       #index=False)

    #создание признаков
    df_features, df_target = create_features.create_features(df_train, target_name=target_name,
                                                             lag_max=lag_max, rolling_step=0, 
                                                             lag_predict=lag_predict,
                                                             db_user=db_user, db_password=db_password, table_name='weekend')

    #разбивка датасета на выборки
    df_features_train, df_features_test = data_split.data_split_handler(df_features, 
                                                                        split_date=split_date,  
                                                                        target_hour=target_hour, 
                                                                        show_info=False)
    
    df_target_train, df_target_test = data_split.data_split_handler(df_target, 
                                                                    split_date=split_date,  
                                                                    target_hour=target_hour, 
                                                                    show_info=False)


    #подготовка модели
    model_cb = create_model.model_catboost(df_features_train, df_target_train,
                                           hyperparams, 
                                           cat_features=cat_features,
                                           verbose=False, 
                                           random_state=25)

    #прогноз
    df_target_train_predict, df_target_test_predict = model_predict.model_predict(model_cb, 
                                                                                  df_features_train, df_target_train,
                                                                                  df_features_test, df_target_test)

    #обратное преобразование Бокса-Кокса
    df_target_train = transform_dataset.boxcox_transform_reverse(df_target_train, const, best_lambda)
    df_target_train_predict = transform_dataset.boxcox_transform_reverse(df_target_train_predict, const, best_lambda)
    df_target_test = transform_dataset.boxcox_transform_reverse(df_target_test, const, best_lambda)
    df_target_test_predict = transform_dataset.boxcox_transform_reverse(df_target_test_predict, const, best_lambda)

    #обработка предиктов
    df_target_train = model_predict.update_values(df_target_train)
    df_target_train_predict = model_predict.update_values(df_target_train_predict)
    df_target_test = model_predict.update_values(df_target_test)
    df_target_test_predict = model_predict.update_values(df_target_test_predict)

    #оценка модели        
    df_eval_hour_total = pd.DataFrame()
    if end_date != split_date:
    
        df_eval = model_eval.model_eval(df_target_train, df_target_train_predict,
                                        df_target_test, df_target_test_predict)
        
        df_eval_total = pd.DataFrame([[f"model_cb_{table_name}_{end_date.strftime('%d%m%Y')}.cbm",
                                       table_name,
                                       date_start_train, date_end_train,
                                       date_start_test, date_end_test,
                                       df_eval.loc['train', 'rmse'], df_eval.loc['train', 'mae'], df_eval.loc['train', 'mape'], df_eval.loc['train', 'r2'],
                                       df_eval.loc['test', 'rmse'], df_eval.loc['test', 'mae'], df_eval.loc['test', 'mape'], df_eval.loc['test', 'r2']
                                       ]],
                                      columns=['model_name',
                                               'techn_proc',
                                               'date_start_train', 'date_end_train',
                                               'date_start_test', 'date_end_test',
                                               'rmse_train', 'mae_train', 'mape_train', 'r2_train',
                                               'rmse_test', 'mae_test', 'mape_test', 'r2_test'])                             
    
        for h in range(24):
            df_eval_hour = model_eval.model_eval(df_target_train.iloc[:, h], df_target_train_predict.iloc[:, h],
                                                 df_target_test.iloc[:, h], df_target_test_predict.iloc[:, h])

            df_ = pd.DataFrame([[f"model_cb_{table_name}_{end_date.strftime('%d%m%Y')}.cbm",
                                 table_name,
                                 date_start_train, date_end_train,
                                 date_start_test, date_end_test,
                                 h,
                                 df_eval_hour.loc['train', 'rmse'], df_eval_hour.loc['train', 'mae'], df_eval_hour.loc['train', 'mape'], df_eval_hour.loc['train', 'r2'],
                                 df_eval_hour.loc['test', 'rmse'], df_eval_hour.loc['test', 'mae'], df_eval_hour.loc['test', 'mape'], df_eval_hour.loc['test', 'r2']
                                 ]],
                                columns=['model_name',
                                         'techn_proc',
                                         'date_start_train', 'date_end_train',
                                         'date_start_test', 'date_end_test',
                                         'hour',
                                         'rmse_train', 'mae_train', 'mape_train', 'r2_train',
                                         'rmse_test', 'mae_test', 'mape_test', 'r2_test'])

            df_eval_hour_total = pd.concat([df_eval_hour_total, df_])
            
    else:
    
        df_eval = model_eval.model_eval(df_target_train, df_target_train_predict,
                                        df_target_test=None, df_target_test_predict=None)
        
        df_eval_total = pd.DataFrame([[f"model_cb_{table_name}_{end_date.strftime('%d%m%Y')}.cbm",
                                       table_name,
                                       date_start_train, date_end_train,
                                       date_start_test, date_end_test,
                                       df_eval.loc['value', 'rmse'], df_eval.loc['value', 'mae'], df_eval.loc['value', 'mape'], df_eval.loc['value', 'r2']
                                       ]],
                                      columns=['model_name',
                                               'techn_proc',
                                               'date_start_train', 'date_end_train',
                                               'date_start_test', 'date_end_test',
                                               'rmse', 'mae', 'mape', 'r2']) 
        for h in range(24):
            df_eval_hour = model_eval.model_eval(df_target_train.iloc[:, h], df_target_train_predict.iloc[:, h])

            df_ = pd.DataFrame([[f"model_cb_{table_name}_{end_date.strftime('%d%m%Y')}.cbm",
                                 table_name,
                                 date_start_train, date_end_train,
                                 date_start_plan, date_end_plan,
                                 h,
                                 df_eval_hour.loc['value', 'rmse'], df_eval_hour.loc['value', 'mae'], df_eval_hour.loc['value', 'mape'], df_eval_hour.loc['value', 'r2']
                                 ]],
                                columns=['model_name',
                                         'techn_proc',
                                         'date_start', 'date_end',
                                         'date_start_plan', 'date_end_plan',
                                         'hour',
                                         'rmse', 'mae', 'mape', 'r2',
                                         ])

            df_eval_hour_total = pd.concat([df_eval_hour_total, df_])
            
    if save_model:
        df_total.to_excel(os.path.join(path, 'clear_datasets', f"clear_dataset_{table_name}_{end_date.strftime('%d%m%Y')}.xlsx"),
                                       index=False)
        df_boxcox.to_excel(os.path.join(path, 'boxcox_params', f"boxcox_params_{table_name}_{end_date.strftime('%d%m%Y')}.xlsx"),
                                        index=False)
        model_cb.save_model(os.path.join(path, 'models', f"model_cb_{table_name}_{end_date.strftime('%d%m%Y')}.cbm"))
        
    df_eval = df_eval.reset_index(drop=True)
    df_eval_hour_total = df_eval_hour_total.reset_index(drop=True)
    
    return df_eval_total, df_eval_hour_total


def make_predicts(path,
                  db_user, db_password, table_name,
                  start_date=None, end_date=None, split_date=None,
                  lag_max=24, lag_predict=24, target_hour=0,
                  n_predict=24, cat_features=None, target_name='fact',
                  anomaly_activate=True,
                  model_eval_flag=True):
    
    
    if start_date is None:
        start_date = pd.to_datetime('today').to_period('M').to_timestamp()
        print(start_date)

    df = load_dataset.load_dataset(file_path=None,
                                   db_user=db_user, db_password=db_password, table_name=table_name,
                                   start_date=pd.to_datetime(start_date) - pd.to_timedelta(lag_max, unit='h'), end_date=end_date,
                                   target_name=target_name)
    
    if split_date is None:
        split_date = df.index.max() + pd.to_timedelta(1, unit='h')
    
    #очистка от аномалий
    df_total = pd.read_excel(os.path.join(path, 'clear_datasets', f"clear_dataset_{table_name}_{start_date.strftime('%d%m%Y')}.xlsx"))
    df, df_total = detect_anomaly.detect_anomaly(df, target_name=target_name, 
                                                 file_path=None, db_user=db_user, db_password=db_password,
                                                 group_list=['weekend', 'month', 'hour'], 
                                                 k_value=1.5, agg='median', 
                                                 df_total=df_total,
                                                 show_info=False,
                                                 activate=anomaly_activate,
                                                 this_year=False)
    
    #преобразование Бокса-Кокса
    df_boxcox = pd.read_excel(os.path.join(path, 'boxcox_params', f"boxcox_params_{table_name}_{start_date.strftime('%d%m%Y')}.xlsx"))
    const = df_boxcox['const'].iloc[0]
    best_lambda = df_boxcox['best_lambda'].iloc[0]
    
    df, const, best_lambda = transform_dataset.boxcox_transform_direct(df, target_name=target_name, 
                                                                       const=const, best_lambda=best_lambda,
                                                                       plot_kde=False, show_info=False)
    
    #создание признаков
    df_features, df_target = create_features.create_features(df, target_name=target_name,
                                                             lag_max=lag_max, rolling_step=0, 
                                                             lag_predict=lag_predict,
                                                             db_user=db_user, db_password=db_password, table_name='weekend')

    #разбивка датасета на выборки
    df_features, df_features_ = data_split.data_split_handler(df_features, 
                                                              split_date=split_date,  
                                                              target_hour=target_hour, 
                                                              show_info=False)
    df_target, df_target_ = data_split.data_split_handler(df_target, 
                                                          split_date=split_date,  
                                                          target_hour=target_hour, 
                                                          show_info=False)

    
    #подготовка модели
    model_cb = CatBoostRegressor()
    model_cb.load_model(os.path.join(path, 'models', f"model_cb_{table_name}_{start_date.strftime('%d%m%Y')}.cbm"))
    
    df_target_predict, _ = model_predict.model_predict(model_cb, 
                                                       df_features, df_target,
                                                       df_features_, df_target_)
    
    #обратное преобразование Бокса-Кокса
    df_target = transform_dataset.boxcox_transform_reverse(df_target, const, best_lambda)
    df_target_predict = transform_dataset.boxcox_transform_reverse(df_target_predict, const, best_lambda)

    #обработка предиктов
    df_target = model_predict.update_values(df_target)
    df_target_predict = model_predict.update_values(df_target_predict)

    #оценка модели
    if model_eval_flag:
        df_eval = model_eval.model_eval(df_target, df_target_predict)
        print(df_eval)
    
    df_target_predict = pd.DataFrame(df_target_predict.values.flatten(),
                                     columns=['predict'], 
                                     index=pd.date_range(start=df_features.index.min(), end=df_features.index.max()+pd.to_timedelta(n_predict-1, unit='h'), 
                                                         freq='h'))
    
    
    #создание признаков для прогноза на будущее
    end_date = df_target_predict.index.max() + pd.to_timedelta(1, unit='h')
    print(f'end_date = {end_date}')
    
    #создание признаков для прогноза
    df_next = pd.concat([df[end_date - pd.to_timedelta(n_predict, unit='h'):end_date - pd.to_timedelta(1, unit='h')], 
                         pd.DataFrame([], index=[end_date])
                        ])
    df_next_features = create_features.create_features(df_next, target_name=target_name,
                                                       lag_max=lag_max, rolling_step=0, 
                                                       lag_predict=None,
                                                       db_user=db_user, db_password=db_password, table_name='weekend')
    
    #подготовка датасета для прогноза 
    df_next_target = pd.concat([df_target_, pd.DataFrame(index=[end_date])])
    
    df_next_target_predict, _ = model_predict.model_predict(model_cb, 
                                                            df_next_features, df_next_target,
                                                            df_features_, df_target_)

    #обратное преобразование Бокса-Кокса
    df_next_target_predict = transform_dataset.boxcox_transform_reverse(df_next_target_predict, const, best_lambda)

    #обработка предиктов
    df_next_target_predict = model_predict.update_values(df_next_target_predict)
    df_next_target_predict = pd.DataFrame(df_next_target_predict.values.flatten(),
                                     columns=['predict'], 
                                     index=pd.date_range(start=df_next_features.index.min(), end=df_next_features.index.max()+pd.to_timedelta(n_predict-1, unit='h'), 
                                                         freq='h'))

    return  df_target_predict, df_next_target_predict
    
                    
def update_techn_proc_hparams(df,
                              conn_string, table_name='techn_proc_model_hparams', 
                              column_name='predict',
                              show_info=False,
                              activate=True): 
    
    df_ = df.copy()
    
    with pyodbc.connect(conn_string) as conn:
        with conn.cursor() as cur:
            for row in df_.itertuples(name=None, index=False):
                for i in range(len(row)):
                    query = f'''update {table_name}
                                set {df_.columns[i]} = '{row[i]}'
                                where model_name = '{row[0]}'
                                  and depth = {row[6]}
                                  and iter = {row[7]}
                             '''
                    if show_info:
                        print(query)
                    if activate:
                        cur.execute(query)
                        cur.commit()
                    
                    
def update_techn_proc_fact(df,
                           conn_string, table_name, 
                           show_info=False,
                           activate=True):
        
    df_ = df.copy()
    
    with pyodbc.connect(conn_string) as conn:
        with conn.cursor() as cur:
            for row in df_.itertuples(name=None, index=True):
                query = f'''update {table_name}
                            set fact = {row[1]}
                            where date_start = '{row[0]}'
                         '''
                if show_info:
                    print(query)
                if activate:
                    cur.execute(query)
                    cur.commit()
                            
def update_techn_proc_predict(df,
                              conn_string, table_name,
                              show_info=False,
                              activate=True):
            
    df_ = df.copy()
    
    with pyodbc.connect(conn_string) as conn:
        with conn.cursor() as cur:
            for row in df_.itertuples(name=None, index=True):
                query = f'''update {table_name}
                            set predict = {row[1]}
                            where date_start = '{row[0]}'
                         '''
                if show_info:
                    print(query)
                if activate:
                    cur.execute(query)
                    cur.commit()
                    
def update_techn_proc_scores(df,
                             conn_string, table_name,
                             show_info=False,
                             activate=True):
    
    df_ = df.copy()  
    
    with pyodbc.connect(conn_string) as conn:
        with conn.cursor() as cur:
            for row in df_.itertuples(name=None, index=False):
                for i in range(len(row)):
                    query = f'''update {table_name}
                                set {df_.columns[i]} = '{row[i]}'
                                where model_name = '{row[0]}'
                                  and hour = '{row[6]}'
                             '''
                    if show_info:
                        print(query)
                    if activate:
                        cur.execute(query)
                        cur.commit()




