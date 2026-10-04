import pandas as pd
from pathlib import Path

base_dir = Path(__file__).resolve().parent
input_path = base_dir / 'data.csv'
output_path = base_dir / 'data_clean.csv'

df = pd.read_csv(input_path)

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

df['engine_hp'] = df['engine_hp'].fillna(df['engine_hp'].median())
df['engine_cylinders'] = df['engine_cylinders'].fillna(df['engine_cylinders'].median())
df['number_of_doors'] = df['number_of_doors'].fillna(df['number_of_doors'].mode()[0])
df['engine_fuel_type'] = df['engine_fuel_type'].fillna(df['engine_fuel_type'].mode()[0])
df['market_category'] = df['market_category'].fillna('Unknown')

df = df.copy()

for col in ['year', 'engine_hp', 'engine_cylinders', 'number_of_doors', 'highway_mpg', 'city_mpg', 'popularity', 'msrp']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Reemplaza valores vacíos restantes por la media de la columna para asegurar integridad
for col in df.columns:
    if df[col].isna().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# Guarda el dataset limpio
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f'Dataset limpio guardado en: {output_path}')
print(f'NaN totales: {df.isna().sum().sum()}')
print(df.isna().sum()[df.isna().sum() > 0])
