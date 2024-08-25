import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


def model_eval(df_target_train, df_target_train_predict,
               df_target_test=None, df_target_test_predict=None):
    
    df_target_train_ = df_target_train.copy()
    df_target_train_predict_ = df_target_train_predict.copy()
        
    mae_train = round(mean_absolute_error(df_target_train_, df_target_train_predict_), 2)
    mape_train = round(mean_absolute_percentage_error(df_target_train_+1, df_target_train_predict_+1), 2) #делаем +1, так как метрика MAPE очень чувствительная к значения близким к нулю
    rmse_train = round(mean_squared_error(df_target_train_, df_target_train_predict_, squared=False), 2)
    r2_train = round(r2_score(df_target_train_.T, df_target_train_predict_.T), 2) #для r2_score данные предварительно трансформируем
    
    if df_target_test is not None and df_target_test_predict is not None:
        df_target_test_ = df_target_test.copy()
        df_target_test_predict_ = df_target_test_predict.copy()
    
        mae_test = round(mean_absolute_error(df_target_test_, df_target_test_predict_), 2)
        mape_test = round(mean_absolute_percentage_error(df_target_test_+1, df_target_test_predict_+1), 2)
        rmse_test = round(mean_squared_error(df_target_test_, df_target_test_predict_, squared=False), 2)
        r2_test = round(r2_score(df_target_test_.T, df_target_test_predict_.T), 2)
        
        df_metrics = pd.DataFrame([[mae_train, mape_train, rmse_train, r2_train], 
                                   [mae_test, mape_test, rmse_test, r2_test]], 
                                  columns=['mae', 'mape', 'rmse', 'r2'],
                                  index=['train', 'test'])
    
        return df_metrics
    else:
        df_metrics = pd.DataFrame([[mae_train, mape_train, rmse_train, r2_train]], 
                                  columns=['mae', 'mape', 'rmse', 'r2'],
                                  index=['value'])
        
        return df_metrics
    


if __name__ == "__main__":
    df_metrics = model_eval(df_target_train, df_target_train_predict,
                            df_target_test=None, df_target_test_predict=None)







