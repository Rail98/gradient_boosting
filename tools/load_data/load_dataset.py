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
        
def load_dataset(file_path=None, 
                 db_user=None, db_password=None, table_name=None,
                 start_date=None, end_date=None,
                 target_name='fact'):
                 
    if file_path is not None:
        df_ = pd.read_excel(file_path, parse_dates=[0]) #, index_col=[0])
    elif db_user is not None and db_password is not None: #подтягиваем данные из базы (если есть доступ к нему)
        with pyodbc.connect(db_pglshp_parameters(db_user, db_password)) as conn:
            df_ = pd.read_sql(f'''SELECT [date_start],
                                         [fact]
                                  FROM [ITPerfomanceMetrics].[dbo].{table_name}
                                  WHERE fact IS NOT NULL''', 
                              conn) 
                                 
    df_.columns = ['datetime', target_name]
    df_.index = df_['datetime']
    df_ = df_.sort_index()
    
    if start_date is not None and end_date is None:
        df_ = df_[start_date:]
    elif start_date is None and end_date is not None:
        df_ = df_[:end_date]
    elif start_date is not None and end_date is not None:
        df_ = df_[start_date:end_date]
        
    return df_[[target_name]]


if __name__ == "__main__":
    file_path = os.path.join('datasets', 'tp1.xlsx')
    df = load_dataset(file_path, target_name='fact', start_date='2021-01-01', end_date='2024-05-01')
    print(df)





