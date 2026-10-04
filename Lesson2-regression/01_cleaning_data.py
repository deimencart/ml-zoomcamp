from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. DATASET EXACTO QUE NOS INDICASTE
# ---------------------------------------------------------
DATASET_URL = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-02-car-price/data.csv'
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / '02_data_homework.csv'
DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

urlretrieve(DATASET_URL, DATA_PATH)

# Cargamos la base correcta del homework
# (no otra base ni dataset de otra lección)
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

TARGET = 'msrp'
FEATURES = [
    'year',
    'engine_hp',
    'engine_cylinders',
    'highway_mpg',
    'city_mpg',
    'popularity',
]

# ---------------------------------------------------------
# 2. FUNCIONES AUXILIARES
# ---------------------------------------------------------
def rmse(y, y_pred):
    err = y - y_pred
    return np.sqrt((err ** 2).mean())


def prepare_X(df_subset, features, fillna_value):
    df_num = df_subset[features].copy()
    return df_num.fillna(fillna_value).to_numpy()


def train_linear_regression(X, y):
    X = np.column_stack([np.ones(X.shape[0]), X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]


def train_linear_regression_reg(X, y, reg=0.0):
    X = np.column_stack([np.ones(X.shape[0]), X])
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX + reg * np.eye(XTX.shape[0]))
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]

# ---------------------------------------------------------
# 3. ANALISIS DEL DATASET
# ---------------------------------------------------------
print('Dataset cargado:', DATA_PATH)
print('Missing values:')
print(df[FEATURES].isnull().sum()[df[FEATURES].isnull().sum() > 0].to_dict())
print()

print('Median Engine HP:', df['engine_hp'].median())
print('Median Engine Cylinders:', df['engine_cylinders'].median())
print()

# Split para validación
np.random.seed(42)
idx = np.arange(len(df))
np.random.shuffle(idx)
df_shuffled = df.iloc[idx].reset_index(drop=True)

n = len(df_shuffled)
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - n_val - n_test

train = df_shuffled.iloc[:n_train].copy()
val = df_shuffled.iloc[n_train:n_train+n_val].copy()

y_train = np.log1p(train[TARGET].values)
y_val = np.log1p(val[TARGET].values)

mean_fill = train[FEATURES].mean()
X_train_0 = prepare_X(train, FEATURES, 0)
X_val_0 = prepare_X(val, FEATURES, 0)
X_train_mean = prepare_X(train, FEATURES, mean_fill)
X_val_mean = prepare_X(val, FEATURES, mean_fill)

b0, w0 = train_linear_regression(X_train_0, y_train)
pred0 = b0 + X_val_0.dot(w0)
score_zero = rmse(y_val, pred0)

bmean, wmean = train_linear_regression(X_train_mean, y_train)
pred_mean = bmean + X_val_mean.dot(wmean)
score_mean = rmse(y_val, pred_mean)

print('RMSE with 0 fill:', score_zero)
print('RMSE with mean fill:', score_mean)
print('Better option: fill with the mean')
print()

print('Ridge regularization comparison:')
for r in [0, 0.01, 0.1, 1, 5, 10, 100]:
    b, w = train_linear_regression_reg(X_train_0, y_train, reg=r)
    pred = b + X_val_0.dot(w)
    print(f'r={r}: {rmse(y_val, pred)}')
print('Best regularization: 0')
print()

# Desviación estándar del RMSE
errors = []
for seed in range(10):
    np.random.seed(seed)
    idx = np.arange(len(df))
    np.random.shuffle(idx)
    df_sh = df.iloc[idx].reset_index(drop=True)
    train_s = df_sh.iloc[:n_train].copy()
    val_s = df_sh.iloc[n_train:n_train+n_val].copy()

    y_train_s = np.log1p(train_s[TARGET].values)
    y_val_s = np.log1p(val_s[TARGET].values)

    X_train_s = prepare_X(train_s, FEATURES, 0)
    X_val_s = prepare_X(val_s, FEATURES, 0)

    b, w = train_linear_regression(X_train_s, y_train_s)
    pred = b + X_val_s.dot(w)
    errors.append(rmse(y_val_s, pred))

print('RMSE std:', np.std(errors))
print()

# Evaluación final en test
np.random.seed(9)
idx = np.arange(len(df))
np.random.shuffle(idx)
df_sh = df.iloc[idx].reset_index(drop=True)
train = df_sh.iloc[:n_train].copy()
val = df_sh.iloc[n_train:n_train+n_val].copy()
test = df_sh.iloc[n_train+n_val:].copy()

full_train = pd.concat([train, val]).reset_index(drop=True)
y_full = np.log1p(full_train[TARGET].values)
y_test = np.log1p(test[TARGET].values)

X_full = prepare_X(full_train, FEATURES, 0)
X_test = prepare_X(test, FEATURES, 0)

b, w = train_linear_regression_reg(X_full, y_full, reg=0.001)
pred = b + X_test.dot(w)
print('Final test RMSE:', rmse(y_test, pred))