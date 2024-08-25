#!/usr/bin/env python
# coding: utf-8

# ### Импорт библиотек

# In[1]:


#импортитуем библиотеки
import os #для управления файлами 
import time #для фиксирования времени выполнения кода
import pandas as pd #для обработки данных
import numpy as np #для математических преобразований (например, округление, найти логарифм числа и т.д.)
import matplotlib.pyplot as plt #визуализация данных
import seaborn as sns #визуализация данных

from statsmodels.tsa.seasonal import seasonal_decompose #показать тренд/сезонность/шумы
from statsmodels.tsa.stattools import adfuller #провести тест Дики-Фуллера на станциорнасть ряда

from scipy.stats import boxcox #преобразование Бокса-Кокса, чтобы нормализовать данные (приблизить к станционарности)

from sklearn.model_selection import train_test_split #разбивка данных на тренировочную/валидационную/тестовую выборки
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score #метрики качества данных
from sklearn.tree import DecisionTreeRegressor #модель Дерево решения
from sklearn.ensemble import RandomForestRegressor #модель Случайного леса 
from catboost import CatBoostRegressor #Градиентный бустинг


# In[2]:


#отключить предупрежения
import warnings
warnings.filterwarnings("ignore")


# ### Загрузка данных

# Cоздадим функцию по подгрузке данных **`load_dateset(file_name, start_date)`**, где 
# 
# `file_name` - название excel-файла, в котором содержатся исторические данные;
# 
# `start_date` - начало периода выгрузки (указываем, если нужно усечь часть данных).
# 
# Пример содержания файла:
# ![image-2.png](attachment:image-2.png)
# **Имя полей можно указать любым (необязательно datetime или oper_num)*

# In[3]:


def load_dataset(file_path, target_name='fact', start_date=None):
    df_ = pd.read_excel(file_path, parse_dates=[0], index_col=[0])
    df_.columns = [target_name]
    df_ = df_.sort_index()
    
    if start_date is not None:
        df_ = df_[start_date:]

    return df_


# In[4]:


file_path = os.path.join('datasets', 'tp1.xlsx')
df = load_dataset(file_path, target_name='fact', start_date='2021-01-01')
df


# ### Визуализация и оценка данных

# Создадим функцию для визуализации и оценки на станционарность наших данных. От этого будет зависеть качество будущей модели. Оценку проведем с помощью теста Дикки-Фуллера

# In[5]:


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


# In[6]:


visual_dataset(df, target_name='fact', rolling_step=24*7)


# In[7]:


def adf_test(df, target_name='fact', rolling_step=24*7):
    df_ = df.copy()

    #проводим тест Дики-Фуллера на станционарность ряда (важно для модели)
    adf_test = adfuller(df_[target_name])
    print('p-value = ' + str(adf_test[1]))
    print('Если p-value < 5%, то можно предположить, что ряд станционарен.')


# In[8]:


adf_test(df, target_name='fact', rolling_step=24*7)


# ### Прямое и обратное преобразование Бокса-Кокса

# Улучшить качество прогноза можно с помощью преобразования Бокса-Кокса. Данный метод позволит снизить влияние дисперсии и приблизить распределение ряда к нормальному виду:
# ![image.png](attachment:image.png)

# In[9]:


#прямое преобразование Бокса-Кокса
def boxcox_transform_direct(df, target_name='fact', plot_kde=True):
    df_ = df.copy()
    
    #для того, чтобы применить преобразование Бокса-Кокса требуется, чтобы входные данные были положительными (> 0)
    const = abs(df_[target_name].min()) + 1
    df_[target_name] += const
    #воспользуемся методом boxcox из пакета scipy.stats
    transformed_data, best_lambda = boxcox(df_[target_name])
    df_ = pd.DataFrame(transformed_data, columns=[target_name], index = df_.index)
    
    if plot_kde == True:
        sns.histplot(transformed_data, kde=True).set_title("График KDE")
        plt.show()
    
    #важно возвращать переменные const и best_lambda для обратного преобразования Бокса-Кокса
    print(f'const = {const}')
    print(f'best_lambda = {best_lambda}')
    return df_, const, best_lambda


# In[10]:


df, const, best_lambda = boxcox_transform_direct(df, target_name='fact', plot_kde=True)
df


# Формула обратного преобразования Бокса-Кокса
# ![image.png](attachment:image.png)

# In[11]:


#обратное преобразование Бокса-Кокса
def boxcox_transform_reverse(df, const, best_lambda):
    df_ = df.copy()
    if best_lambda != 0:
        df_ = (1 + best_lambda*df_)**(1/best_lambda) - const
    else:
        df_ = np.exp(df_)
    return df_


# In[12]:


boxcox_transform_reverse(df, const, best_lambda)


# ### Разность временного ряда

# Вычислив разность временного ряда, мы можем привести ряд к станционарному виду. Обычно для этого достаточно применить разность первого порядка. Если это не поможет, то нужно применить разность второго порядка - разность разностей :)
# 
# Разность можно найти по-разному. Например, найти между значениями с прошлым показателем, либо с указанным шагом. Так, например, разность с шагом 24 будет <u>дневной</u> разностью, а с 24×7 - <u>недельной</u>.

# In[13]:


#находим разность первого порядка
def calculate_diff1(df, target_name='fact', 
                    fact_lag=24):
    df_ = df.copy()
    df_[f'{target_name}_lag'] = df_[target_name].shift(fact_lag)
    df_['diff1'] = df_[target_name] - df_[f'{target_name}_lag']
    df_ = df_.dropna()
    return df_[[target_name]], df_[[f'{target_name}_lag']], df_[['diff1']]


# In[14]:


#разность планирую применить для временных рядов  с явно выраженным трендом
fact_lag = 24
df_diff1 = calculate_diff1(df, target_name='fact', 
                           fact_lag=fact_lag)


# In[15]:


df_diff1[0].head(3)


# In[16]:


df_diff1[1].head(3)


# In[17]:


df_diff1[2].head(3)


# ### Исключение выбросов (аномалий)

# Проведем чистку датасета от аномальных значений, которые искажают действительные значения ряда. Для этого требуется выполнить следущие шаги:
# - подготовить датасет (найдем календарные дни, вычислим праздничные дни);
# - сгруппировать данные по каждому месяцу, часу и по признаку выходной/рабочий день;
# - из полученной группировки найти среднее значение, медиану, кванитили;
# - соединить полученные выборки с исходными данными (метод `merge`);
# - построить диаграмму размаха (ящик с усами) по каждому часу и определим границы усов;
# - если фактическое значение выходит за границу, то заменить на выбранную переменную (среднее или медиану).

# In[18]:


#подгружаем выходные дни из БД или через excel-файл
def db_pglshp_parameters(db_user, db_password):
        server = 'p-glshp-db01'
        database = 'ITPerfomanceMetrics'
        driver = '{SQL Server Native Client 11.0}'
        conn_string = f'SERVER={server};DATABASE={database};DRIVER={driver};UID={user};PWD={password}'
        return conn_string
    
def load_weekend(df, file_path=None, db_user, db_password):
    df_ = df.copy()
    
    #если будет указан путь до файла с выходными днями, то подгружаем оттуда
    if file_path is not None:
        df_weekend = pd.read_excel(file_path)
    else: #иначе тянем данные из базы (если есть доступ к нему)
        with pyodbc.connect(db_pglshp_parameters(db_user, db_password)) as conn:
            df_weekend = pd.read_sql('''SELECT [weekend],
                                        FROM [ITPerfomanceMetrics].[dbo].[weekend]''', 
                                     conn)
            
    #объединяем df_ и df_weekend с помощю метода merge (аналог join в sql)
    df_['datetime'] = df_.index
    df_['date'] = pd.to_datetime(df_.index.date)
    df_ = df_.merge(df_weekend, left_on='date', right_on='weekend', how='left')
    
    #присвоем столбцу значение 1, если дата попадает на выходной день, и 0 - рабочий день
    df_.loc[~df_['weekend'].isna(), 'weekend'] = 1
    df_['weekend'] = df_['weekend'].fillna(0)
    
    df_.index = df_['datetime']
    
    return df_


# In[19]:


load_weekend(df, file_path='Нерабочие дни.xlsx')


# In[20]:


#поиск и исключение аномалий
def clear_anomaly(df, target_name='fact', 
                  group_list=['weekend', 'month', 'hour'], 
                  k_value=1.5, agg='median', 
                  show_info=True, 
                  activate=True):  
    df_ = df.copy()
    
    #подготовка данных
    df_ = load_weekend(df_, file_path='Нерабочие дни.xlsx')
    df_['quarter'] = df_.index.quarter
    df_['month'] = df_.index.month
    df_['week'] = df_.index.isocalendar().week
    df_['weekday'] = df_.index.weekday
    df_['hour'] = df_.index.hour
    
    #чтобы определить, аномальное ли значение или нет, найдем медиану на конкретную месяц-день-час
    df_q25 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.25).rename(columns={target_name: 
                                                                                                   f'{target_name}_q25'})
    df_q75 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.75).rename(columns={target_name: 
                                                                                                   f'{target_name}_q75'})
    df_q2 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.02).rename(columns={target_name: 
                                                                                                  f'{target_name}_q2'})
    df_q98 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.98).rename(columns={target_name: 
                                                                                                   f'{target_name}_q98'})
    df_q1 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.01).rename(columns={target_name: 
                                                                                                  f'{target_name}_q1'})
    df_q99 = df_.groupby(group_list, as_index=False)[target_name].quantile(q=0.99).rename(columns={target_name: 
                                                                                                   f'{target_name}_q99'})
    df_count = df_.groupby(group_list, as_index=False)[target_name].count().rename(columns={target_name: 
                                                                                            f'{target_name}_count'})
    df_mean = df_.groupby(group_list, as_index=False)[target_name].mean().rename(columns={target_name: 
                                                                                          f'{target_name}_mean'})
    df_median = df_.groupby(group_list, as_index=False)[target_name].median().rename(columns={target_name: 
                                                                                              f'{target_name}_median'})
    df_total = df_count.merge(df_mean, how='outer').merge(df_median, how='outer')
    df_total = df_total.merge(df_q25, how='outer').merge(df_q75, how='outer')
    df_total = df_total.merge(df_q2, how='outer').merge(df_q98, how='outer')
    df_total = df_total.merge(df_q1, how='outer').merge(df_q99, how='outer')
    df_ = df_.merge(df_total, on=group_list, how='left')
    df_.index = df_['datetime']
    df_
    
    #описательная информация по поиску аномалий
    if show_info == True:
        print(df_[f'{target_name}_count'].value_counts().sort_index()) #кол-во отобранных записей по группировке
        k = 1.5
        anomaly_count = len(df_[(df_[target_name] > df_[f'{target_name}_q75'] + k*(df_[f'{target_name}_q75'] - df_[f'{target_name}_q25']))]) \
                      + len(df_[(df_[target_name] < df_[f'{target_name}_q25'] - k*(df_[f'{target_name}_q75'] - df_[f'{target_name}_q25']))])
        print(f"Количество выбросов по диаграмму размаха при k=1.5: {anomaly_count} ({round(anomaly_count/len(df_)*100, 2)}%)")
        k = 1.2
        anomaly_count = len(df_[(df_[target_name] > df_[f'{target_name}_q75'] + k*(df_[f'{target_name}_q75'] - df_[f'{target_name}_q25']))]) \
                      + len(df_[(df_[target_name] < df_[f'{target_name}_q25'] - k*(df_[f'{target_name}_q75'] - df_[f'{target_name}_q25']))])
        print(f"Количество выбросов по диаграмму размаха при k=1.2: {anomaly_count} ({round(anomaly_count/len(df_)*100, 2)}%)")
        anomaly_count = len(df_[df_[target_name] > df_[f'{target_name}_q99']]) + len(df_[df_[target_name] < df_[f'{target_name}_q1']])
        print(f"Количество выбросов от 1% до 99% квантиля: {anomaly_count} ({round(anomaly_count/len(df_)*100, 2)}%)")
        anomaly_count = len(df_[df_[target_name] > df_[f'{target_name}_q98']]) + len(df_[df_[target_name] < df_[f'{target_name}_q2']])
        print(f"Количество выбросов от 5% до 95% квантиля: {anomaly_count} ({round(anomaly_count/len(df_)*100, 2)}%)")
        
    #ощищаем датасет от аномалий
    if activate == True:
        def clear(row, k):
            if row[target_name] < row[f'{target_name}_q25'] - k*(row[f'{target_name}_q75'] - row[f'{target_name}_q25']):
                return round(row[f'{target_name}_{agg}'])
            elif row[target_name] > row[f'{target_name}_q75'] + k*(row[f'{target_name}_q75'] - row[f'{target_name}_q25']):
                return round(row[f'{target_name}_{agg}'])
            else:
                return row[target_name]
        df_[target_name] = df_.apply(clear, k=k_value, axis=1)
        
    return df_[[target_name]]


# In[21]:


df = clear_anomaly(df, target_name='fact', 
                   group_list=['weekend', 'month', 'hour'], 
                   k_value=1.5, agg='median', 
                   show_info=True, 
                   activate=True)


# ### Создание признаков

# In[22]:


def create_features(df, target_name='fact',
                    lag_max=0, rolling_step=0, 
                    lag_predict=1):
    df_ = df.copy()
    
    #выходные дни
    df_ = load_weekend(df, file_path='Нерабочие дни.xlsx')
    
    #календарные дни
    df_['quarter'] = df_.index.quarter
    df_['month'] = df_.index.month
    df_['week'] = df_.index.isocalendar().week
    df_['weekday'] = df_.index.weekday
    df_['hour'] = df_.index.hour
    
    #lag_max
    for lag in range(1, lag_max+1):
        df_[f'lag_{lag}'] = df_['fact'].shift(lag) 
    
    #rolling_step
    if rolling_step > 1:
        df_['rolling_mean'] = df_[target_name].shift().rolling(rolling_step).mean()
        df_['rolling_std'] = df_[target_name].shift().rolling(rolling_step).std()
           
    #predict_lag
    column_predict = []
    for lag in range(0, lag_predict):
        df_[f'{target_name}_{lag}'] = df_[target_name].shift(-lag)
        column_predict.append(f'{target_name}_{lag}')
    
    df_ = df_.drop(columns=['fact', 'datetime', 'date'])
    df_ = df_.dropna()
    
    df_features = df_.drop(columns=column_predict)
    df_target = df_[column_predict]
    
    return df_features, df_target


# In[23]:


df_features, df_target = create_features(df, target_name='fact',
                                         lag_max=24, rolling_step=0, 
                                         lag_predict=24) #предсказываем на день вперед


# In[24]:


df_features.head(3)


# In[25]:


df_target.head(3)


# ### Разбивка данных на обучающую, валидационную и тестовую выборки

# Рассмотрим несколько вариантов определения выборок:
# 
#     - указать самостоятельно по дате;
#     - разделить в % соотношении (например, 3:1:1);
#     - накопительная выборка.

# Примечание*
# 
# Оценку модели предпочтительнее проводить по тому времени, когда будет производиться прогноз.
# Планируется запускать **ежедневный** прогноз каждый день в **00:00:00**, поэтому приотитнее оценка в этот промежуток времени.
# Плюсом также явдяется то, что у нас достаточное кол-во записей для оценки модели.

# In[26]:


#самостоятельно
def tts_handler(df, 
                start_date_test, hour=None, 
                show_info=True):
    df_ = df.copy()
    df_train = df_[:start_date_test][:-1] #[:-1]исключаем послденюю запись, так как она попадает на тестовую выборку
    df_test = df_[start_date_test:]
    
    #оставляем записи, в промежутке которого будет выполняться прогноз
    if hour is not None:
        df_train = df_train[df_train.index.hour == 0]
        df_test = df_test[df_test.index.hour == 0]
        
    #показать размер выборок
    if show_info == True:
        print(f'Размер обучающей выборки: {df_train.shape} ({round(len(df_train)/(len(df_train) + len(df_test))*100, 2)}% от общей выборки)')
        print(f'Размер тестовой выборки: {df_test.shape} ({round(len(df_test)/(len(df_train) + len(df_test))*100, 2)}% от общей выборки)')
    return df_train, df_test


# In[27]:


start_date_test = '2024-01-01 00:00:00'
df_features_train, df_features_test = tts_handler(df_features, 
                                                  start_date_test, hour=0, 
                                                  show_info=True)
df_target_train, df_target_test = tts_handler(df_target, 
                                              start_date_test, hour=0, 
                                              show_info=False)


# ### Построение модели

# In[28]:


#Градиентный бустинг
def model_catboost(df_features_train, df_target_train,
                   hyperparams, 
                   verbose=True, 
                   random_state=25):
    start_ = time.time() 
    model = CatBoostRegressor(**hyperparams, 
                                random_state=random_state, 
                                verbose=verbose
                                )
    model.fit(df_features_train, df_target_train, verbose=verbose)
    end_ = time.time()

    dt_ = round(end_-start_, 1)
    print(f'Время выполнения функции {dt_} сек.')
    
    return model


# In[29]:


hyperparams = {'loss_function': 'MultiRMSE', 
               'iterations': 100, #100 
               'learning_rate': 0.03, #0.03
               'depth': 6, #6
               }
model_cb = model_catboost(df_features_train, df_target_train,
                          hyperparams, 
                          verbose=True, 
                          random_state=25)


# ### Прогноз модели

# In[30]:


def model_predict(model, 
                  df_features_train, df_target_train,
                  df_features_test, df_target_test):

    target_train_predict = model.predict(df_features_train) #прогноз на обучающей выборки
    target_test_predict = model.predict(df_features_test) #прогноз на тестовой выборки
    df_target_train_predict = pd.DataFrame(target_train_predict, columns=df_target_train.columns, index=df_target_train.index)
    df_target_test_predict = pd.DataFrame(target_test_predict, columns=df_target_test.columns, index=df_target_test.index)
    
    return df_target_train_predict, df_target_test_predict


# In[32]:


df_target_train_predict, df_target_test_predict = model_predict(model_cb, 
                                                                df_features_train, df_target_train,
                                                                df_features_test, df_target_test)


# In[33]:


df_target_train_predict.head(3)


# In[34]:


df_target_test_predict.head(3)


# In[35]:


#обработка предиктов
def update_values(df):
    df_ = df.copy()
    
    
    def upd(row):
        if row < 0:
            return 0 #замена отрицательных значений на ноль
        return round(row) #округление до целого
    df_ = df_.applymap(upd)
    
    return df_


# ### Оценка модели

# In[36]:


def model_eval(df_target_train, df_target_train_predict,
               df_target_test, df_target_test_predict):
    
    df_target_train_ = df_target_train.copy()
    df_target_test_ = df_target_test.copy()
    df_target_train_predict_ = df_target_train_predict.copy()
    df_target_test_predict_ = df_target_test_predict.copy()
        
    mae_train = round(mean_absolute_error(df_target_train_, df_target_train_predict_), 2)
    mae_test = round(mean_absolute_error(df_target_test_, df_target_test_predict_), 2)
    mape_train = round(mean_absolute_percentage_error(df_target_train_+1, df_target_train_predict_+1), 2) #делаем +1, так как метрика MAPE очень чувствительная к значения близким к нулю
    mape_test = round(mean_absolute_percentage_error(df_target_test_+1, df_target_test_predict_+1), 2)
    mse_train = round(mean_squared_error(df_target_train_, df_target_train_predict_), 2)
    mse_test = round(mean_squared_error(df_target_test_, df_target_test_predict_), 2)
    r2_train = round(r2_score(df_target_train_.T, df_target_train_predict_.T), 2) #для r2_score данные предварительно трансформируем
    r2_test = round(r2_score(df_target_test_.T, df_target_test_predict_.T), 2)
    
    df_metrics = pd.DataFrame([[mae_train, mape_train, mse_train, r2_train], 
                               [mae_test, mape_test, mse_test, r2_test]], 
                              columns=['mae', 'mape', 'mse', 'r2'],
                              index=['train', 'test'])
    
    return df_metrics


# In[37]:


#применяем обратное преобразование Бокса-Кокса
df_target_train_boxcox_reverse = boxcox_transform_reverse(df_target_train, const, best_lambda)
df_target_train_predict_boxcox_reverse = boxcox_transform_reverse(df_target_train_predict, const, best_lambda)
df_target_test_boxcox_reverse = boxcox_transform_reverse(df_target_test, const, best_lambda)
df_target_test_predict_boxcox_reverse = boxcox_transform_reverse(df_target_test_predict, const, best_lambda)


# In[38]:


#ощищаем предикты от отрицательных значений, и округляем их
df_target_train_boxcox_reverse = update_values(df_target_train_boxcox_reverse)
df_target_train_predict_boxcox_reverse = update_values(df_target_train_predict_boxcox_reverse)
df_target_test_boxcox_reverse = update_values(df_target_test_boxcox_reverse)
df_target_test_predict_boxcox_reverse = update_values(df_target_test_predict_boxcox_reverse)


# In[39]:


#оценим нашу модель
model_eval(df_target_train_boxcox_reverse, df_target_train_predict_boxcox_reverse,
           df_target_test_boxcox_reverse, df_target_test_predict_boxcox_reverse)


# ### Запись в базу данных

# In[40]:


#...


# In[ ]:




