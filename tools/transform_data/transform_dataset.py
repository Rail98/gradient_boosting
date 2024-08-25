import pandas as pd
import numpy as np
from scipy.stats import boxcox
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


#прямое преобразование Бокса-Кокса
def boxcox_transform_direct(df, target_name='fact', const=None, best_lambda=None, plot_kde=True, show_info=True):
    df_ = df.copy()
    
    #для того, чтобы применить преобразование Бокса-Кокса требуется, чтобы входные данные были положительными (> 0)
    if const is None:
        const = abs(df_[target_name].min()) + 1
        df_[target_name] += const
        
    #воспользуемся методом boxcox из пакета scipy.stats
    if best_lambda is not None:
        if best_lambda != 0:
            df_[target_name] = (df_[target_name]**best_lambda - 1)/best_lambda
        else:
            df_[target_name] = np.log(df_[target_name])
    else:
        transformed_data, best_lambda = boxcox(df_[target_name])
        df_ = pd.DataFrame(transformed_data, columns=[target_name], index = df_.index)
    
    if plot_kde == True:
        sns.histplot(transformed_data, kde=True).set_title("График KDE")
        plt.show()
    
    if show_info == True:
        print(f'const = {const}')
        print(f'best_lambda = {best_lambda}')
    
    #важно возвращать переменные const и best_lambda для обратного преобразования Бокса-Кокса
    return df_, const, best_lambda


#обратное преобразование Бокса-Кокса
def boxcox_transform_reverse(df, const, best_lambda):
    df_ = df.copy()
    
    if best_lambda != 0:
        df_ = (1 + best_lambda*df_)**(1/best_lambda) - const
    else:
        df_ = np.exp(df_)
        
    return df_


#разность первого порядка
def diff1_transform(df, target_name='fact', 
                    lag_diff=24):
    df_ = df.copy()
    
    df_[f'{target_name}_lag'] = df_[target_name].shift(lag_diff)
    df_['diff1'] = df_[target_name] - df_[f'{target_name}_lag']
    
    df_ = df_.dropna()
    return df_[[target_name]], df_[[f'{target_name}_lag']], df_[['diff1']]
                           
                           
if __name__ == "__main__":
    #прямое преобразование Бокса-Кокса
    df, const, best_lambda = boxcox_transform_direct(df, target_name='fact', 
                                                     plot_kde=True, show_info=True)
    print(df)                           
    
    #обратное преобразование Бокса-Кокса
    df = boxcox_transform_reverse(df, const, best_lambda)
    print(df)

    #разность первого порядка
    df_diff1 = diff1_transform(df, target_name='fact', 
                               lag_diff=24)
    print(df_diff1)


