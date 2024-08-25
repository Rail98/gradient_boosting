#импорт модулей
import pandas as pd
import pyodbc
from threading import Thread
import os

from dit_techn_proc_func import (db_parameters_glshp, db_parameters_ods, 
                                 unload_data, update_techn_proc_fact)
from env import params

import warnings
warnings.filterwarnings('ignore')


#период выгрузки
end = pd.to_datetime('today').floor('h') #окончание
end_tz_3h = pd.to_datetime('today').floor('h') - pd.to_timedelta(3, unit='h')
start = end - pd.to_timedelta(24, unit='h') #начало
start_tz_3h = start - pd.to_timedelta(3, unit='h')

end = "'" + end.strftime('%Y-%m-%d %H:%M:%S') + "'"
end_tz_3h = "'" + end_tz_3h.strftime('%Y-%m-%d %H:%M:%S') + "'"
start = "'" + start.strftime('%Y-%m-%d %H:%M:%S') + "'"
start_tz_3h = "'" + start_tz_3h.strftime('%Y-%m-%d %H:%M:%S') + "'"

print(f'start_date = {start}')
print(f'end_date = {end}')
print(f'start_date_tz_3h = {end_tz_3h}')
print(f'end_date_tz_3h = {end_tz_3h}')

#константы
path_sql = r'C:\Users\z-gabdulkhakovrf\Desktop\Python_scripts\ITPerfomanceMetrics\Techn_proc\sql'

#соединение с базой ods
db_user_ods = params['db_user_ods']
db_password_ods = params['db_password_ods']
conn_string_ods = db_parameters_ods(db_user_ods, db_password_ods)

#glshp
db_user_glshp = params['db_user_glshp']
db_password_glshp = params['db_password_glshp']
conn_string_glshp = db_parameters_glshp(db_user_glshp, db_password_glshp)


# Запускаем код
#список техпроцессов, по которым предоставили данные
#второй техпроцесс запускается отдельно, так как считается по-другому
print(f'start = {start}')
print(f'end = {end}')
print()
for n in range(1, 14):
    
    table_name = f'techn_proc_{n}' 
    column_name = 'fact'
    
    def thread_func(n, table_name, column_name):
    
        file_name = f'tp{n}.sql'
        file_path = os.path.join(path_sql, file_name)
        
        try:    
            with open(file_path, 'r') as file:
                query = file.read()

            #запуск функций
            if n in [1, 6, 11]:
                query = query.replace('@end', end_tz_3h).replace('@start', start_tz_3h)
                df = unload_data(conn_string_ods,
                                 query,
                                 add_hour=3)
            elif n in [4]:
                query = query.replace('@end', end).replace('@start', start)
                df = unload_data(conn_string_glshp,
                                 query,
                                 add_hour=None,
                                 oper_id=False)
            else:
                query = query.replace('@end', end).replace('@start', start)
                df = unload_data(conn_string_ods,
                                 query,
                                 add_hour=None)

            update_techn_proc_fact(df,
                                   conn_string_glshp, table_name, 
                                   show_info=False,
                                   activate=True)

            print(f'Данные в таблицу {table_name} успешно записаны')
            
        except:
            print(f'Ошибка при загрузке данных в таблицу {table_name}')
    
    
    thread = Thread(target=thread_func, args=(n, table_name, column_name))
    thread.start()
    
thread.join()


#print('Успех. Все данные записаны в базу')
#import time
#time.sleep(10)

