import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# 1️⃣ SIMULACIÓN DE DATOS
# ==============================

np.random.seed(42)

# Generar 500 registros de tráfico
fechas = [datetime(2025, 3, np.random.randint(1, 31), np.random.randint(6, 22)) for _ in range(500)]
nivel_congestion = np.random.randint(1, 10, size=500)
velocidad_promedio = np.random.randint(10, 80, size=500)
condiciones_climaticas = np.random.choice(['Soleado', 'Lluvioso', 'Nublado'], size=500)

# Creación del DataFrame
df = pd.DataFrame({
    'fecha_hora': fechas,
    'nivel_congestion': nivel_congestion,
    'velocidad_promedio': velocidad_promedio,
    'condiciones_climaticas': condiciones_climaticas
})

# Introducir valores nulos y duplicados para prueba de limpieza
df.loc[10, 'nivel_congestion'] = np.nan
df = pd.concat([df, df.iloc[0:5]], ignore_index=True)

# ==============================
# 2️⃣ LIMPIEZA Y TRANSFORMACIÓN
# ==============================

# Eliminar duplicados y valores nulos
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Convertir fecha en formato estándar
df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])

# Extraer la hora para análisis y modelado
df['hora'] = df['fecha_hora'].dt.hour

# Guardar los datos limpios
df.to_csv('datos_trafico_limpios.csv', index=False)
print("✅ Datos simulados y limpiados correctamente.")

# ==============================
# 3️⃣ ANÁLISIS DESCRIPTIVO
# ==============================

# Histograma de niveles de congestión
plt.figure(figsize=(8,5))
sns.histplot(df['nivel_congestion'], bins=10, kde=True, color='blue')
plt.title('Distribución de Congestión Vehicular')
plt.xlabel('Nivel de Congestión')
plt.ylabel('Frecuencia')
plt.savefig("histograma_congestion.png")
plt.show()

# Boxplot de velocidad según clima
plt.figure(figsize=(8,5))
sns.boxplot(x=df['condiciones_climaticas'], y=df['velocidad_promedio'])
plt.title('Distribución de Velocidad por Condición Climática')
plt.xlabel('Condición Climática')
plt.ylabel('Velocidad Promedio (km/h)')
plt.savefig("boxplot_velocidad.png")
plt.show()

# ==============================
# 4️⃣ MODELADO PREDICTIVO
# ==============================

# Convertir variables categóricas en numéricas (One-Hot Encoding)
df = pd.get_dummies(df, columns=['condiciones_climaticas'], drop_first=True)

# Definir variables predictoras (X) y variable objetivo (y)
X = df[['velocidad_promedio', 'hora', 'condiciones_climaticas_Lluvioso', 'condiciones_climaticas_Nublado']]
y = df['nivel_congestion']

# División en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# ==============================
# 5️⃣ EVALUACIÓN DEL MODELO
# ==============================

# Predicción en datos de prueba
y_pred = modelo.predict(X_test)

# Cálculo de métricas de error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"📊 Resultados del Modelo:")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")

# Gráfico de predicciones vs valores reales
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Comparación de Predicciones vs Valores Reales')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
plt.savefig("grafico_predicciones.png")
plt.show()