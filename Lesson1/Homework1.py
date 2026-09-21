import pandas as pd
import numpy as np

route = "Lesson1/Datasets/HW1/car_fuel_efficiency_2026.csv"
try: 
    df = pd.read_csv(route)
    print("DataFrame loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print(f"File not found at {route}. Please check the file path.")

print (df.info())

#cuantos tipos hay en esta etiquieta
print(df['fuel_type'].nunique())

#cuantos datos faltan.

columnas_con_nulos = df.isnull().any().sum()
print(f"Columnas con valores faltantes: {columnas_con_nulos}")

# Filtrar por origen 'Asia' y obtener el máximo de la columna 'fuel_efficiency_mpg'
max_eficiencia_asia = df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()

print(f"The maximum fuel efficiency of cars from Asia is: {max_eficiencia_asia}")

# Median value of horesepower for 

median_original = df ['horsepower'].median()
print(f"The median horsepower of the original dataset is: {median_original}")

mode = df['horsepower'].mode()[0]
print(f"The mode of horsepower is: {mode}")

df['horsepower'] = df['horsepower'].fillna(mode)

nueva_mediana = df['horsepower'].median()
print(f"4. Nueva mediana de horsepower: {nueva_mediana}")


# linear regression

# 1. Seleccionar todos los coches de Asia (limpiando posibles espacios en blanco)
asia_cars = df[df['origin'].str.strip() == 'Asia']

# 2. Seleccionar solo las columnas 'vehicle_weight' y 'model_year'
columnas_seleccionadas = asia_cars[['vehicle_weight', 'model_year']]

# 3. Seleccionar los primeros 7 valores
primeros_7 = columnas_seleccionadas.head(7)

# 4. Obtener el array de NumPy subyacente (X)
X = primeros_7.to_numpy()

# 5. Calcular la multiplicación matricial entre la transpuesta de X y X (XTX)
XTX = X.T @ X

# 6. Invertir la matriz XTX
XTX_inv = np.linalg.inv(XTX)

# 7. Crear el array y con los valores indicados
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# 8. Multiplicar la inversa de XTX por la transpuesta de X, y luego por y (w)
w = XTX_inv @ X.T @ y

# 9. Calcular la suma de todos los elementos del resultado w
suma_pesos_final = w.sum()

print(f"El resultado de la suma de los elementos de w es: {suma_pesos_final}")


# print(pd.__version__)
