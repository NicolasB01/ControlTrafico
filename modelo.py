import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Cargar datos limpios
df = pd.read_csv("datos_trafico_limpios.csv")

# Convertir variables categóricas (clima) en numéricas
df = pd.get_dummies(df, columns=['condiciones_climaticas'], drop_first=True)

# Selección de variables predictoras y variable objetivo
X = df[['velocidad_promedio', 'hora', 'condiciones_climaticas_Lluvioso', 'condiciones_climaticas_Nublado']]
y = df['nivel_congestion']

# División de datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento del modelo Random Forest
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Predicción y evaluación del modelo
y_pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"Error Absoluto Medio (MAE): {mae}")
print(f"Error Cuadrático Medio (MSE): {mse}")