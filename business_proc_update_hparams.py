import time
import pandas as pd
import pyodbc

from env import params
from dit_techn_proc_func import (db_parameters_glshp, calculate_model, make_predicts,
                                 update_techn_proc_hparams)

#константы
path = r'C:\Users\z-gabdulkhakovrf\Desktop\Python_scripts\ITPerfomanceMetrics\Techn_proc\saved_parameters'

#glshp
db_user_glshp = params['db_user_glshp']
db_password_glshp = params['db_password_glshp']
conn_string_glshp = db_parameters_glshp(db_user_glshp, db_password_glshp)

#подбор гиперпараметров
depth = [2, 4, 6] #глубина модели
iterations = [200, 400, 600, 800, 1000, 1200] #кол-во итераций

end_date = pd.to_datetime('today').to_period('M').to_timestamp() #дата создания модели
column_list = ['model_name', 'techn_proc', 
               'date_start_train', 'date_end_train',
               'date_start_test', 'date_end_test', 
               'depth', 'iter',
               'rmse_train', 'mae_train', 'mape_train', 'r2_train', 
               'rmse_test', 'mae_test', 'mape_test', 'r2_test',
               'time'] 


#techn_proc_model_hparams
for n in range(1, 14):
    table_name = f'techn_proc_{n}'
    print('------------')
    print(table_name)
    print()

    for d in depth:
        for i in iterations:
            print(f'depth = {d}')
            print(f'iterations = {i}')
            hyperparams = {'loss_function': 'MultiRMSE', 
                           'iterations': i,
                           'depth': d,
                           }
            
            try:
                time_start = time.time()
                df_eval, _ = calculate_model(path, hyperparams, 
                                             db_user_glshp, db_password_glshp, table_name,
                                             )
                time_end = int(time.time() - time_start)
                
                df_eval['depth'] = d
                df_eval['iter'] = i
                df_eval['time'] = time_end
                
                df_eval = df_eval[column_list]
                update_techn_proc_hparams(df_eval,
                                          conn_string_glshp, table_name='techn_proc_model_hparams', 
                                          column_name='predict',
                                          show_info=False,
                                          activate=True)
                print('Данные записаны в базу')
                
            except:
                print('Ошибка при поиске гиперпараметров')
                continue

            print()
            
        
    print('------------')
    print()
    

#techn_proc_model_hparams_best
for n in range(1, 14):
    table_name = f'techn_proc_{n}'
    model_name = f"model_cb_{table_name}_{end_date.strftime('%d%m%Y')}.cbm"
    query = f'''DELETE FROM techn_proc_model_hparams_best
                WHERE model_name = '{model_name}';
                
                INSERT INTO techn_proc_model_hparams_best
                SELECT TOP 1 * 
                FROM techn_proc_model_hparams (NOLOCK) 
                WHERE model_name = '{model_name}'
                  AND mape_test IS NOT NULL
                ORDER BY mape_test, depth, iter
                '''
    print(query)
    try:
        with pyodbc.connect(conn_string_glshp) as conn:
            cur = conn.cursor()
            cur.execute(query)
            cur.commit()
        print(f'Успех. Данные в таблицу {table_name} записаны')        
    except:
        print(f'Запись в таблицу {table_name} не была выполнена')
