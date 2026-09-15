import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# insert the dataset , this dataset is taken from kaggle named heeraldedhia/bike-buyers
# the dataset contains 1000 samples and 13  column 
df = pd.read_csv('bike_buyers.csv')
print("Dataset Shape:", df.shape)
# remove id column
df_clean = df.drop(columns=['ID']).copy()

# missing value correction n median
num_cols = ['Income', 'Children', 'Cars', 'Age']
for col in num_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Categorical features Impute with Mode
cat_cols = ['Marital Status', 'Gender', 'Home Owner']
for col in cat_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

# encode target variable bike purchased (used as input feature now)
df_clean['Purchased Bike'] = df_clean['Purchased Bike'].map({'Yes': 1, 'No': 0})

# target is income for reg
X = df_clean.drop(columns=['Income'])
y = df_clean['Income'].values.reshape(-1, 1)

# Feature Encoding and Scaling lists
categorical_features = ['Marital Status', 'Gender', 'Education', 'Occupation', 'Home Owner', 'Commute Distance', 'Region', 'Purchased Bike']
numerical_features = ['Children', 'Cars', 'Age']

ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ]
)

X_processed = ct.fit_transform(X)

# Train test split (80-20 Split)
X_train, X_test, y_train_raw, y_test_raw = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# scale continuous target variable income or gradient descent will blow up
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train_raw)
y_test = sc_y.transform(y_test_raw)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Processed Features Shape:", X_processed.shape[1])

class LinearRegression:
    def __init__(self, lr=0.01, ep=1500):
        self.lr = lr
        self.ep = ep
        self.w = None
        self.b = None
        self.loss_hist = []

    def fit(self, X, y):
        n, f = X.shape
        self.w = np.zeros((f, 1))
        self.b = 0.0

        for i in range(self.ep):
            pred = np.dot(X, self.w) + self.b
            err = pred - y
            
            # MSE loss
            l = np.mean(err ** 2) / 2.0
            self.loss_hist.append(l)
            
            dw = (1 / n) * np.dot(X.T, err)
            db = (1 / n) * np.sum(err)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    # mlp reg model scratch implementation
class MLPRegression:
    def __init__(self, layers, lr=0.01, ep=1500):
        self.layers = layers
        self.lr = lr
        self.ep = ep
        self.loss_hist = []
        
        # init params with he method
        np.random.seed(42)
        self.weights = []
        self.biases = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def d_relu(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        acts = [X]
        zs = []
        curr = X
        
        # hidden layers with relu
        for i in range(len(self.weights) - 1):
            z = np.dot(curr, self.weights[i]) + self.biases[i]
            zs.append(z)
            curr = self.relu(z)
            acts.append(curr)
            
        # last output layer linear activation for reg
        z_out = np.dot(curr, self.weights[-1]) + self.biases[-1]
        zs.append(z_out)
        acts.append(z_out)
        return acts, zs

    def backward(self, X, Y, acts, zs):
        n = X.shape[0]
        dw_list = [None] * len(self.weights)
        db_list = [None] * len(self.biases)
        
        # output error
        dz = acts[-1] - Y
        dw_list[-1] = (1 / n) * np.dot(acts[-2].T, dz)
        db_list[-1] = (1 / n) * np.sum(dz, axis=0, keepdims=True)
        
        # backprop hidden layers
        for l in range(len(self.weights) - 2, -1, -1):
            da = np.dot(dz, self.weights[l+1].T)
            dz = da * self.d_relu(zs[l])
            dw_list[l] = (1 / n) * np.dot(acts[l].T, dz)
            db_list[l] = (1 / n) * np.sum(dz, axis=0, keepdims=True)
            
        return dw_list, db_list

    def fit(self, X, Y):
        t0 = time.time()
        for _ in range(self.ep):
            acts, zs = self.forward(X)
            pred = acts[-1]
            
            # mse loss calc
            l = np.mean((pred - Y) ** 2) / 2.0
            self.loss_hist.append(l)
            
            # gradient steps
            dw_list, db_list = self.backward(X, Y, acts, zs)
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * dw_list[i]
                self.biases[i] -= self.lr * db_list[i]
                
        self.t_train = time.time() - t0

    def predict(self, X):
        acts, _ = self.forward(X)
        return acts[-1]

# train mlp model
mlp_model = MLPRegression(layers=[X_train.shape[1], 32, 16, 1], lr=0.01, ep=1500)
mlp_model.fit(X_train, y_train)

# mlp testing preds
mlp_p_scaled = mlp_model.predict(X_test)
mlp_p_raw = sc_y.inverse_transform(mlp_p_scaled)

# mlp metrics calculation
mlp_mse_val = np.mean((y_test_raw - mlp_p_raw) ** 2)
mlp_rmse_val = np.sqrt(mlp_mse_val)
mlp_mae_val = np.mean(np.abs(y_test_raw - mlp_p_raw))
mlp_r2_val = 1 - (np.sum((y_test_raw - mlp_p_raw) ** 2) / np.sum((y_test_raw - np.mean(y_test_raw)) ** 2))

print("MLP Reg Results:")
print("MSE:", round(float(mlp_mse_val), 2))
print("RMSE:", round(float(mlp_rmse_val), 2))
print("MAE:", round(float(mlp_mae_val), 2))
print("R2:", round(float(mlp_r2_val), 4))
print("Time taken:", round(mlp_model.t_train, 4), "sec")
