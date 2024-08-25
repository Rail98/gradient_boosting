import pandas as pd 
import pyodbc
import warnings
warnings.filterwarnings("ignore")


def db_pglshp_parameters(db_user, db_password):
        server = 'p-glshp-db01'
        database = 'ITPerfomanceMetrics'
        driver = '{SQL Server Native Client 11.0}'
        conn_string = f'SERVER={server};DATABASE={database};DRIVER={driver};UID={db_user};PWD={db_password}'
        return conn_string
    
def load_weekend(df, 
                 file_path=None, 
                 db_user=None, db_password=None, table_name='weekend'):
    df_ = df.copy()
    
    #если будет указан путь до файла с выходными днями, то подгружаем оттуда
    if file_path is not None:
        df_weekend = pd.read_excel(file_path)
    elif db_user is not None and db_password is not None: #подтягиваем данные из базы (если есть доступ к нему)
        with pyodbc.connect(db_pglshp_parameters(db_user, db_password)) as conn:
            df_weekend = pd.read_sql(f'''SELECT [weekend]
                                         FROM [ITPerfomanceMetrics].[dbo].{table_name}''', 
                                     conn)
    else:
        print('Требуется указать путь до файла с выходными либо логин и пароль к БДшке')
        
    df_weekend.columns = ['weekend']
            
    #объединяем df_ и df_weekend с помощю метода merge (аналог join в sql)
    df_['datetime'] = df_.index
    df_['date'] = pd.to_datetime(df_.index.date)
    df_ = df_.merge(df_weekend, left_on='date', right_on='weekend', how='left')
    
    #присвоем столбцу значение 1, если дата попадает на выходной день, и 0 - рабочий день
    df_.loc[~df_['weekend'].isna(), 'weekend'] = 1
    df_['weekend'] = df_['weekend'].fillna(0)
    
    df_.index = df_['datetime']
    
    return df_.drop(columns=['datetime', 'date'])


if __name__ == "__main__":
    file_path = os.path.join('datasets', 'tp1.xlsx')
    df = load_weekend(df, file_path='Нерабочие дни.xlsx', db_user=None, db_password=None)
    print(df)






