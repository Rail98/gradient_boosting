import pandas as pd
import pyodbc

from env import params
from dit_techn_proc_func import (db_parameters_glshp, calculate_model, make_predicts,
                                 update_techn_proc_predict)

#константы
path = r'C:\Users\z-gabdulkhakovrf\Desktop\Python_scripts\ITPerfomanceMetrics\Techn_proc\saved_parameters' #пцть, где схранены файлы для моделей
#end_date = pd.to_datetime('today').to_period('M').to_timestamp() #дата создания модели

#glshp
db_user_glshp = params['db_user_glshp']
db_password_glshp = params['db_password_glshp']
conn_string_glshp = db_parameters_glshp(db_user_glshp, db_password_glshp)

for n in range(1, 14):
    table_name = f'techn_proc_{n}'
    
    try:
        df_target_predict, df_next_target_predict = make_predicts(path,
                                                                  db_user_glshp, db_password_glshp, table_name,
                                                                  target_name='fact',
                                                                  anomaly_activate=True)

        #update_techn_proc_predict(df_target_predict,
                                  #conn_string_glshp, table_name,
                                  #show_info=True,
                                  #activate=True)

        update_techn_proc_predict(df_next_target_predict,
                                  conn_string_glshp, table_name,
                                  show_info=False,
                                  activate=True)
        print(f'Прогнозы на таблицу {table_name} успешно записаны')
    except:
        print(f'Ошибка при наполнении таблицы {table_name}')
