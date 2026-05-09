import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

#-----------------------------------------------------------
# prédire la résistance d'une bactérie
#-----------------------------------------------------------

## simuler le dataset 

X, y = make_classification(
    n_samples=500,
    n_features = 3,
    n_informative = 3,
    n_redundant = 0,
    n_classes = 2,
    random_state=42
)

# Voici 3 variables pertinentes pour prédire la résistance bactérienne :
# 1. Présence d'un gène de résistance (binaire : 0/1)
# Par exemple mecA pour le SARM, ou blaZ pour la résistance à la pénicilline. C'est souvent le prédicteur le plus direct.

# 2. Épaisseur de la paroi cellulaire (continue, en nm)
# Une paroi plus épaisse (notamment chez les Gram+) réduit la pénétration des antibiotiques, ce qui est un mécanisme physique 
# de résistance bien documenté.

# 3. Taux de mutation du génome (continue, mutations/gène/génération)
# Un taux élevé favorise l'émergence rapide de variants résistants, c'est un indicateur du potentiel évolutif de la souche.

# 1. gene_resistance : binariser avec un seuil (la structure de corrélation avec y est préservée)
gene_resistance = (X[:, 0] > 0).astype(int)

# 2. epaisseur_paroi_nm : ramener en nm positif (ex: entre 10 et 80 nm)
epaisseur = X[:, 1]
epaisseur_paroi_nm = 10 + 70 * (epaisseur - epaisseur.min()) / (epaisseur.max() - epaisseur.min())

# 3. taux_mutation : ramener entre 0.001 et 0.02 (mutations/gène/génération)
mutation = X[:, 2]
taux_mutation = 0.001 + 0.019 * (mutation - mutation.min()) / (mutation.max() - mutation.min())

df = pd.DataFrame({
    "gene_resistance": gene_resistance,
    "epaisseur_paroi_nm": epaisseur_paroi_nm.round(2),
    "taux_mutation": taux_mutation.round(5),
    "resistance": y
})

df.head()

#-------------------------------------------------------------------------------
# visualisations
#-------------------------------------------------------------------------------

#----------------------------------
# boxplot épaisseur VS resistance
#----------------------------------

fig, ax = plt.subplots(figsize = (6, 5))

groups = [
  df[df["resistance"] == 0]["epaisseur_paroi_nm"],
  df[df["resistance"] == 1]["epaisseur_paroi_nm"]
]


ax.boxplot(groups, labels = ["Non résistantes", "Résistantes"])
ax.set_ylabel("Epaisseur paroi (nm)")
ax.set_title("Bactéries")
plt.show()

# On a l'impression que les bactéries résistantes ont une paroi
# en moyenne plus épaisse que les bactéries non résistantes
# ce qui est logique.

#----------------------------------
# boxplot tx_mutation VS resistance
#----------------------------------

fig, ax = plt.subplots()

groups = [
  df[df["resistance"] == 0]["taux_mutation"],
  df[df["resistance"] == 1]["taux_mutation"]
]


ax.boxplot(groups, labels = ["Non résistantes", "Résistantes"])
ax.set_ylabel("Taux mutation (mut/gene/generation")
ax.set_title("Bactéries")
plt.show()

# On a l'impression que les bactéries résistantes ont un
# taux de mutation plus élevé que les bactéries non résistantes
# ce qui est logique.

#-----------------------------------------------
# stacked barplot resistance, gene resistance
#-----------------------------------------------

fig, ax = plt.subplots()

df[["resistance", "gene_resistance"]].groupby(by = "resistance").value_counts()

category_names = ['Absence gène de résistance', 'Présence gène de résistance']
results = {
    'Non résistantes': [221, 28],
    'Résistantes': [212, 39],
}

def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return fig, ax


survey(results, category_names)
plt.show()

# Il y a un tout petit peu plus de bactéries qui ont 
# un gène de résistance dans les bactéries résistantes 
# mais cette variable ne devrait pas peser beaucoup pour prédire
# la résistance d'une bactérie.

#-------------------------------------------------------------------------------
# Régression logistique
#-------------------------------------------------------------------------------

from sklearn.linear_model import LogisticRegression

# split train et test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
# 88% de bonnes prédictions sur l'échantillon test (accuracy)

# coefficients du modèles
for name, coef in zip(df.columns[:-1], model.coef_.flatten()):
    print(f"{name}: {coef:.4f}")

# matrice de confusion
from sklearn import metrics
y_pred = model.predict(X_test)
conf = metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = [0, 1])

cm_display.plot()
plt.show()

# sensibilité et scpécificité
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

sensibilite = tp / (tp + fn)   # recall de la classe positive
specificite = tn / (tn + fp)   # pas dans sklearn directement

print(f"Sensibilité : {sensibilite:.3f}")
print(f"Spécificité : {specificite:.3f}")

# ROC AUC
from sklearn.metrics import RocCurveDisplay

y_score = model.predict_proba(X_test)[:, 1]  # probabilité de la classe positive

display = RocCurveDisplay.from_predictions(
    y_test,
    y_score,
    name="Résistante vs Non résistante",
    plot_chance_level=True,
)

display.ax_.set(
    xlabel="Taux de Faux Positifs (1 - Spécificité)",
    ylabel="Taux de Vrais Positifs (Sensibilité)",
    title="Courbe ROC - Résistance bactérienne"
)

plt.show()

#-------------------------------------------------------------------------------
# optimiser la classification avec l'index de Younden
#-------------------------------------------------------------------------------

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)
youden_index = tpr - fpr
best_threshold = thresholds[youden_index.argmax()]

print(f"Seuil optimal (Youden) : {best_threshold:.3f}")

y_pred_youden = (y_score >= best_threshold).astype(int)

# matrice de confusion
conf = metrics.confusion_matrix(y_test, y_pred_youden)
print(conf)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = [0, 1])

cm_display.plot()
plt.show()

# accuracy
metrics.accuracy_score(y_test, y_pred_youden)
# l'accuracy a un peu augmenté

# sensibilité et scpécificité
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_youden).ravel()

sensibilite = tp / (tp + fn)   
specificite = tn / (tn + fp)   

print(f"Sensibilité : {sensibilite:.3f}")
print(f"Spécificité : {specificite:.3f}")

# la sensibilité à un peu augmenté, avec ce nouvel index, 
# les bactéries vraiment résistantes sont mieux détéctés comme
# étant résistantes
# mais la spécificité à un peu diminué, les bactéries non résistantes
# sont moins bien détéctées.
