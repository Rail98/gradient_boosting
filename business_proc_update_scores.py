import pandas as pd
import pyodbc

from env import params
from dit_techn_proc_func import (db_parameters_glshp, calculate_model, make_predicts,
                                 update_techn_proc_scores)

#константы
path = r'C:\Users\z-gabdulkhakovrf\Desktop\Python_scripts\ITPerfomanceMetrics\Techn_proc\saved_parameters' #путь, где схранены файлы для моделей
end_date = pd.to_datetime('today').to_period('M').to_timestamp() #дата создания модели

#glshp
db_user_glshp = params['db_user_glshp']
db_password_glshp = params['db_password_glshp']
conn_string_glshp = db_parameters_glshp(db_user_glshp, db_password_glshp)

for n in range(1, 14):
    table_name = f'techn_proc_{n}'
    model_name = f"model_cb_{table_name}_{end_date.strftime('%d%m%Y')}.cbm"
    query = f'''SELECT * 
                FROM techn_proc_model_hparams_best (nolock) 
                WHERE model_name = '{model_name}'
                '''
    print(table_name)
    
    try:
        with pyodbc.connect(conn_string_glshp) as conn:
            df = pd.read_sql(query, conn)
            
        hyperparams = {'loss_function': 'MultiRMSE', 
                       'iterations': df.loc[0, 'iter'], 
                       #'learning_rate': 0.03, #[0.03, 0.1, 0.3]
                       'depth': df.loc[0, 'depth'], 
                       }
        
        #оценка модели на обучающей и тестовой выборке
        df_eval, df_eval_hour_total = calculate_model(path, hyperparams, 
                                                      db_user_glshp, db_password_glshp, table_name,
                                                      #split_date_flag='equal',
                                                      save_model=False)
        #запись оценки модели на тестовой и обучающей выборке
        update_techn_proc_scores(df_eval_hour_total,
                                 conn_string_glshp, table_name='techn_proc_model_scores_train_test',
                                 show_info=False,
                                 activate=True)
        print(f'Оценки на train и test выборках записаны для модели {table_name}')
        print()
        
        #оценка модели на всем промежутке
        df_eval, df_eval_hour_total = calculate_model(path, hyperparams, 
                                                      db_user_glshp, db_password_glshp, table_name,
                                                      split_date_flag='equal',
                                                      save_model=True)
        #сохранение модели
        update_techn_proc_scores(df_eval_hour_total,
                                 conn_string_glshp, table_name='techn_proc_model_scores',
                                 show_info=False,
                                 activate=True)
        print(f'Оценки на всем промежутке записаны для модели {table_name}')
        
    except:
        print('Ошибка при запуске прогноза')
        
    print('-----------')
    print()
