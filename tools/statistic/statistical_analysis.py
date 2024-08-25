import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")


def visual_dataset(df, target_name='fact', rolling_step=24*7):
    df_plot = df.copy()
    df_plot.index.name = 'Дата'
    
    #график за последние семь дней
    df_plot[-24*7:].plot(figsize=(7, 4), grid=True, title='График за последнюю неделю')
    
    #общий график со скользящим средним (можно увидеть, как менялось среднее значение исходных данных)
    #шаг среднего значения можно выбрать любой (чем больше, тем сглаженне будет линия) 
    df_plot['Скользящее среднее'] = df_plot[target_name].rolling(rolling_step).mean()
    df_plot.plot(figsize=(9, 5), grid=True, title='График за весь период')
    plt.show()
    
    #тренд/сезонность/шум
    decomposed = seasonal_decompose(df_plot[target_name]) 
    plt.figure(figsize=(7, 7))
    plt.subplot(311)
    decomposed.trend.plot(ax=plt.gca(), xlabel='')
    plt.title('Тренд')
    plt.grid()
    plt.subplot(312)
    decomposed.seasonal[:24*7].plot(ax=plt.gca(), xlabel='')
    plt.title('Сезонность')
    plt.grid()
    plt.subplot(313)
    decomposed.resid.plot(ax=plt.gca(), xlabel='')
    plt.title('Шумы')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    #KDE
    sns.histplot(df_plot[target_name], kde=True).set_title("График KDE")
    plt.show()


def adf_test(df, target_name='fact', rolling_step=24*7):
    df_ = df.copy()

    #проводим тест Дики-Фуллера на станционарность ряда (важно для модели)
    adf_test = adfuller(df_[target_name])
    print('p-value = ' + str(adf_test[1]))
    print('Если p-value < 5%, то можно предположить, что ряд станционарен')


if __name__ == "__main__":
    visual_dataset(df, target_name='fact', rolling_step=24*7)
    adf_test(df, target_name='fact', rolling_step=24*7)
    








