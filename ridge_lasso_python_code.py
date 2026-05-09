import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

#-----------------------------------------------------------
# prédire le prix d'une maison
#-----------------------------------------------------------

## simuler le dataset 

# 500 maisons, 10 features, seulement 5 sont vraiment utiles
X, y, coef_vrais = make_regression(
    n_samples=500,
    n_features=10,
    n_informative=5,   # seulement 5 features informatives !
    noise=30,
    coef=True,
    random_state=42
)

feature_names = [
    "surface_m2", "nb_pieces", "age_maison",
    "dist_centre", "etage", "bruit_1",
    "bruit_2", "bruit_3", "bruit_4", "bruit_5"
]
df = pd.DataFrame(X, columns=feature_names)
df["prix"] = y

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalisation (OBLIGATOIRE pour Ridge/Lasso/ElasticNet)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(df.head())

#---------------------------------------------------------
# régression Ridge
#---------------------------------------------------------

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

# Entraîne Ridge avec alpha=1.0
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_train)
y_pred_ridge = ridge.predict(X_test_sc)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge   = r2_score(y_test, y_pred_ridge)

# Affiche les coefficients
for name, coef in zip(feature_names, ridge.coef_):
    print(f"{name}: {coef}")
    
# Avec Ridge, les coefficients bruit_* sont réduits vers 0
# MAIS jamais exactement à 0 → Ridge garde toutes les features

# alpha fort (100) → coefficients très petits, underfitting
# alpha faible (0.001) → proche de la régression linéaire classique

# RidgeCV cherche automatiquement le meilleur alpha
ridgecv = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
ridgecv.fit(X_train_sc, y_train)
print(f"Meilleur alpha : {ridgecv.alpha_}") # ici il vaut 1

#-------------------------------------------------------
# régression Lasso
#-------------------------------------------------------

from sklearn.linear_model import Lasso, LassoCV

lasso = Lasso(alpha=1.0)
lasso.fit(X_train_sc, y_train)
y_pred_lasso = lasso.predict(X_test_sc)

rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso   = r2_score(y_test, y_pred_lasso)

# Features sélectionnées (coef != 0)
features_selectionnees = [
    name for name, coef in zip(feature_names, lasso.coef_)
    if coef != 0
]
print("Features gardées:", features_selectionnees)
print("Coefficients:")
for name, coef in zip(feature_names, lasso.coef_):
    statut = "✓" if coef != 0 else "✗ éliminée"
    print(f"  {name:15s}: {coef:8.3f}  {statut}")

# Avec alpha=1.0, Lasso devrait garder ~5 features
# (les 5 vraiment informatives) et mettre les 5 bruit_* à 0

# Lasso fait de la sélection de features intégrée :
# les features inutiles = coefficient EXACTEMENT à 0
# → modèle plus interprétable et moins de surapprentissage

# alpha trop grand → Lasso élimine des features importantes
# → underfitting, mauvaises prédictions
# Utiliser LassoCV pour trouver le bon alpha :
lassocv = LassoCV(cv=5, random_state=42)
lassocv.fit(X_train_sc, y_train)
print(f"Meilleur alpha : {lassocv.alpha_:.4f}")
 
#------------------------------------------------------------------
# régression ElasticNet
#------------------------------------------------------------------

from sklearn.linear_model import ElasticNet, ElasticNetCV

enet = ElasticNet(alpha=1.0, l1_ratio=0.5)
enet.fit(X_train_sc, y_train)
y_pred_enet = enet.predict(X_test_sc)

rmse_enet = np.sqrt(mean_squared_error(y_test, y_pred_enet))
r2_enet   = r2_score(y_test, y_pred_enet)

# Comparaison finale des 3 modèles
print("=" * 40)
print(f"{'Modèle':<12} {'RMSE':>8} {'R²':>8}")
print("-" * 40)
print(f"{'Ridge':<12} {rmse_ridge:>8.2f} {r2_ridge:>8.3f}")
print(f"{'Lasso':<12} {rmse_lasso:>8.2f} {r2_lasso:>8.3f}")
print(f"{'ElasticNet':<12} {rmse_enet:>8.2f} {r2_enet:>8.3f}")

# Auto-tuning ElasticNet
enetcv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 1.0],
    cv=5, random_state=42
)
enetcv.fit(X_train_sc, y_train)
print(f"\nMeilleur alpha   : {enetcv.alpha_:.4f}")
print(f"Meilleur l1_ratio: {enetcv.l1_ratio_}")
