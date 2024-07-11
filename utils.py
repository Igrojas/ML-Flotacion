import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import altair as alt
import warnings

warnings.filterwarnings('ignore')

def ProcesarDatos(data):
    
    data.fillna(0, inplace=True)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_prepared = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)
    
    X = pd.get_dummies(data_prepared)
    X.drop(['Recuperación Planta'], inplace=True, axis=1)
    y = data['Recuperación Planta']
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
    
    return X_train, X_test, y_train, y_test


def ComparacionDeModelos(X_train, y_train ):
    models = []
    models.append(('LR', LinearRegression()))
    models.append(('LASSO', Lasso()))
    models.append(('RIDGE', Ridge()))
    models.append(('ELTN', ElasticNet()))
    models.append(('DTR', DecisionTreeRegressor()))
    models.append(('RFR', RandomForestRegressor()))
    models.append(('ABR', AdaBoostRegressor()))
    models.append(('SGDR', SGDRegressor()))
    models.append(('GBR', GradientBoostingRegressor()))
    models.append(('XGB', XGBRegressor()))
    models.append(('CBR', CatBoostRegressor()))

    results = []
    names = []
    scoring = 'neg_mean_squared_error'

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        rmse = np.sqrt(-cv_results)  # Calculate RMSE from MSE
        for value in rmse:
            results.append((name, value))

    results_df = pd.DataFrame(results, columns=['Model', 'RMSE'])
    results_df['Model'] = pd.Categorical(results_df['Model'], categories=[model[0] for model in models], ordered=True)

    return results_df
