# DATA SELECTION, ANALYSIS AND PREPROCESSING
# Dataset used with House Prices is pending for features Kaggle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns



# LOAD DATASET is pending for determante features
# revisar porque no me coge ruta relativa
# File train.csv downloaded from Kaggle recogido de moodle
#*****************************************

df = pd.read_csv("C:\\Users\\victo\\Documents\\masterciberseguridad\\secondyear\\Neuronal\\Code\\train.csv")

print("Dataset head:", df.shape)
print(df.head())

# MISSING VALUES & BASIC ANALYSIS
# ************
print("\nMissing values column:")
missing = df.isnull().sum()#add null values ​​per column
print(missing[missing > 0].sort_values(ascending=False))

print("\nDataset info:")
df.info() #display in table form


# Separate INPUT (X) and OUTPUT (y)
# Output must be numerical real numerical
# ******************
y = df["SalePrice"]
X = df.drop(columns=["SalePrice"])

# Identify categorical and numerical features. Review this item
# ****************************
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nCategorical columns:", len(categorical_cols))
print("Numerical columns:", len(numerical_cols))

# Detect outliers (IQR method) for the report
# *********************************
def detect_outliers_IQR(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (column < lower) | (column > upper)

outliers = detect_outliers_IQR(y)
print("\nOutliers in SalePrice:", outliers.sum())

# Optional graph of SalePrice or other values distribution
plt.figure(figsize=(7,4))
sns.histplot(y, kde=True)
plt.title("SalePrice Distribution")
plt.show()

# ----------------------------------------------------------
# 6. Preprocessing pipeline
# Includes:
# - Imputation (handled internally by OneHotEncoder & StandardScaler)
# - Encoding categorical variables
# - Normalizing numerical variables
# ----------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Train–Test Split (80% – 20%) with shuffle=True the same training in imagenes de 
# tratamiento de imagenes de las practicas de ia
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    shuffle=True,
    random_state=42
)

print("\nTrain size:", X_train.shape)
print("Test size:", X_test.shape)

# Fit preprocessing pipeline
# **************************
X_train_pre = preprocess.fit_transform(X_train)#Aprende estadísticos, categorías, valores para imputación y tranforma los datos
X_test_pre  = preprocess.transform(X_test)#normalización con la media y std aprendidas ycodificación con categorías encontradas tatos de test

print("\nShape after preprocessing:")
print("X_train_pre:", X_train_pre.shape)
print("X_test_pre:", X_test_pre.shape)

# Save the processed datasets (required in the assignment)
# the first have to be 80% and other 20%
# **************************
np.save("X_train_preprocessed.npy", X_train_pre)
np.save("X_test_preprocessed.npy", X_test_pre)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("\nSuccessfully!")
entrada = input("END")
