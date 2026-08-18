import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
roc_auc_score, average_precision_score)


#--------------------------------------------------------------------------
# simulation du dataset de fraude bancaire
#--------------------------------------------------------------------------

X, y = make_classification(
    n_samples=10_000,
    n_features=20,
    n_informative=12,
    n_redundant=4,
    n_repeated=0,
    n_classes=2,
    weights=[0.95, 0.05],   # 95% légitimes, 5% fraudes
    flip_y=0.01,
    random_state=42
)

feature_names = [
    "montant", "heure_jour", "type_marchand", "age_client",
    "nb_trans_7j", "moy_montant_30j", "dist_domicile",
    "freq_pays_etranger", "nb_refus_recents", "solde_moyen",
    "anciennete_compte", "nb_cartes", "score_risque",
    "nb_litiges", "canal_paiement", "devise", "categorie",
    "heure_derniere_trans", "delta_montant", "flag_nuit"
]


df = pd.DataFrame(X, columns = feature_names)
df["fraude"] = y
df.head()
df.describe()

print(f"Transactions : {len(df):,}")
print(f"Fraudes: {y.sum():,} ({y.mean()*100:.1f}%)")

#-------------------------------------------------------------------------------
# échantillon test et echantillon d'apprentissage
#-------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y,
random_state = 42)

#-------------------------------------------------------------------------------
# 1) XGBoost
#-------------------------------------------------------------------------------

from xgboost import XGBClassifier

# scale_pos_weight compense le déséquilibre des classes :
# ratio = nb_négatifs / nb_positifs
scale = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight : {scale:.1f}")

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,      # aussi appelé "eta"
    max_depth=5,
    subsample=0.8,           # fraction de lignes par arbre
    colsample_bytree=0.8,    # fraction de features par arbre
    reg_alpha=0.1,           # L1 (Lasso)
    reg_lambda=1.0,          # L2 (Ridge)
    scale_pos_weight=scale,  # gère le déséquilibre
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42
)

xgb.fit(X_train, y_train, 
eval_set = [(X_test, y_test)], verbose = 50)

y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
auc_xgb  = roc_auc_score(y_test, y_proba_xgb)
pr_xgb   = average_precision_score(y_test, y_proba_xgb)
print(f"\nXGBoost → AUC-ROC : {auc_xgb:.4f}  |  PR-AUC : {pr_xgb:.4f}")

#-------------------------------------------------------------------------------
# lightGBM la vitesse par histogramme
#-------------------------------------------------------------------------------

from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,            # illimité, contrôlé par num_leaves
    num_leaves=31,           # complexité de l'arbre (défaut=31)
    min_child_samples=20,    # min exemples par feuille
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    is_unbalance=True,       # équivalent de scale_pos_weight auto
    random_state=42,
    verbose=-1               # supprime les logs verbeux
)

lgbm.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        # early stopping si pas d'amélioration sur 30 rounds
        __import__('lightgbm').early_stopping(30, verbose=False),
        __import__('lightgbm').log_evaluation(50)
    ]
)

y_proba_lgbm = lgbm.predict_proba(X_test)[:, 1]
auc_lgbm = roc_auc_score(y_test, y_proba_lgbm)
pr_lgbm  = average_precision_score(y_test, y_proba_lgbm)
print(f"\nLightGBM → AUC-ROC : {auc_lgbm:.4f}  |  PR-AUC : {pr_lgbm:.4f}")
print(f"Meilleure itération : {lgbm.best_iteration_}")

# 1. Level-wise : tous les nœuds d'un niveau sont splittés
#    → arbre symétrique, convergence plus stable
#    Leaf-wise : on splitte TOUJOURS la feuille avec le plus grand gain
#    → arbres asymétriques, meilleure réduction de la loss, mais
#    → risque d'overfitting si num_leaves trop grand

# 2. Avec leaf-wise, max_depth seul ne contrôle pas la complexité :
#    un arbre peut être très profond d'un côté et très court de l'autre
#    num_leaves = nombre max de feuilles terminales = contrôle direct
#    Règle empirique : num_leaves < 2^max_depth

# 3. Early stopping surveille la métrique sur le eval_set :
#    si pas d'amélioration pendant N rounds → arrêt automatique
#    → évite l'overfitting, réduit le temps d'entraînement
#    → donne le best_iteration_ : nombre optimal d'arbres


#-------------------------------------------------------------------------------
# Catboost pour variables catégorielles
#-------------------------------------------------------------------------------

from catboost import CatBoostClassifier, Pool

# Simulons quelques vraies features catégorielles
df_train = pd.DataFrame(X_train, columns=feature_names)
df_test  = pd.DataFrame(X_test,  columns=feature_names)

# On convertit 3 features en catégories (comme en vrai)
for col in ["type_marchand", "canal_paiement", "devise"]:
    df_train[col] = pd.cut(
        df_train[col], bins=5,
        labels=["A","B","C","D","E"]
    ).astype(str)
    df_test[col] = pd.cut(
        df_test[col], bins=5,
        labels=["A","B","C","D","E"]
    ).astype(str)

cat_features = ["type_marchand", "canal_paiement", "devise"]

train_pool = Pool(df_train, y_train, cat_features=cat_features)
test_pool  = Pool(df_test,  y_test,  cat_features=cat_features)

cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,           # régularisation L2
    auto_class_weights="Balanced",  # gère le déséquilibre
    eval_metric="AUC",
    random_seed=42,
    verbose=50
)

cat.fit(train_pool, eval_set=test_pool, early_stopping_rounds=30)

y_proba_cat = cat.predict_proba(test_pool)[:, 1]
auc_cat = roc_auc_score(y_test, y_proba_cat)
pr_cat  = average_precision_score(y_test, y_proba_cat)
print(f"\nCatBoost → AUC-ROC : {auc_cat:.4f}  |  PR-AUC : {pr_cat:.4f}")

# 1. Target encoding classique : encoder une catégorie par la moyenne
#    de y crée un leakage (y_train "contamine" X_train)
#    CatBoost "ordered encoding" : pour chaque exemple i,
#    n'utilise que les exemples 0..i-1 pour calculer l'encodage
#    → chaque exemple est encodé sans voir sa propre cible

# 2. Oblivious tree : à chaque profondeur, MÊME feature et MÊME seuil
#    pour tous les nœuds du niveau
#    → arbre parfaitement symétrique
#    → prédiction = lookup dans un tableau de 2^depth cases
#    → inférence ultra-rapide (important en production temps réel)

# 3. Quand choisir CatBoost :
#    ✓ Beaucoup de features catégorielles (texte, IDs, codes)
#    ✓ Pas de temps pour pré-traiter les catégories
#    ✓ Inférence rapide requise en production
#    ✓ Dataset de taille moyenne (XGBoost/LightGBM meilleurs > 1M lignes)


