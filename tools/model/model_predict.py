import pandas as pd
import warnings
import numpy as np
warnings.filterwarnings("ignore")


def model_predict(model, 
                  df_features_train, df_target_train,
                  df_features_test, df_target_test
                  ):

    target_train_predict = model.predict(df_features_train) #прогноз на обучающей выборки
    target_test_predict = model.predict(df_features_test) #прогноз на тестовой выборки
    df_target_train_predict = pd.DataFrame(target_train_predict, columns=df_target_train.columns, index=df_target_train.index)
    df_target_test_predict = pd.DataFrame(target_test_predict, columns=df_target_test.columns, index=df_target_test.index)
    
    return df_target_train_predict, df_target_test_predict
    
def update_values(df):

    df_ = df.copy()

    def upd(row):
        if row < 0:
            return 0 #замена отрицательных значений на ноль
        return np.floor(row) #округление до целого
    df_ = df_.applymap(upd)

    return df_

if __name__ == "__main__":
    #model_predict
    df_target_train_predict, df_target_test_predict = model_predict(model_cb, 
                                                                    df_features_train, df_target_train,
                                                                    df_features_test, df_target_test)
    print(df_target_test_predict)
    
    #update_values
    df_target_test_predict = update_values(df_target_test_predict)
    print(df_target_test_predict)






