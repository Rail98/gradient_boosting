import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")


#Градиентный бустинг
def model_catboost(df_features_train, df_target_train,
                   hyperparams,
                   cat_features=None,
                   verbose=True, 
                   random_state=25):
    model = CatBoostRegressor(**hyperparams, 
                                random_state=random_state, 
                                verbose=verbose
                                )
    model.fit(df_features_train, df_target_train, 
              cat_features=cat_features, 
              verbose=verbose) 
    return model
    

if __name__ == "__main__":
    hyperparams = {'loss_function': 'MultiRMSE', 
                   'iterations': 100,
                   'learning_rate': 0.03,
                   'depth': 6,
                   }
    model_cb = model_catboost(df_features_train, df_target_train,
                              hyperparams, 
                              verbose=True, 
                              random_state=25)








