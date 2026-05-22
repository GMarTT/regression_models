import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

n = 300

regions = ["North", "South", "East", "West"]
hospital_type = ["University", "General", "Private"]
antibiotic = ["Fluoroquinolone", "3GC", "Carbapenem"]
years = np.arange(2015, 2025)

data = pd.DataFrame({
    "region": np.random.choice(regions, n),
    "hospital_type": np.random.choice(hospital_type, n),
    "antibiotic": np.random.choice(antibiotic, n),
    "year": np.random.choice(years, n),
    "antibiotic_consumption": np.random.normal(55, 15, n),
    "temperature": np.random.normal(15, 5, n),
    "population_density": np.random.normal(250, 80, n)
})

# Génération d'une résistance AMR corrélée
data["amr_rate"] = (
    0.4 * data["antibiotic_consumption"]
    + 0.3 * data["temperature"]
    + np.random.normal(0, 8, n)
)

# Bornes réalistes
data["amr_rate"] = data["amr_rate"].clip(0, 100)

print(data.head())
print(data.shape)
print(data.describe())

#------------------------------------------------
# 1) Boxplot: taux de resistance VS type hopital
#-------------------------------------------------

sns.set_theme(style = "whitegrid", palette = "pastel")

plt.figure(figsize=(8,5))

sns.boxplot(
    x="hospital_type",
    y="amr_rate",
    data=data
)

plt.title("AMR rate according to hospital type")
plt.xlabel("Hospital type")
plt.ylabel("AMR rate (%)")

sns.despine(offset=10, trim=True)
plt.show()

#------------------------------------------------
# 2) Consommation AB et résistance AB
#-------------------------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8,5))

sns.relplot(x = "antibiotic_consumption", 
            y = "amr_rate", 
            hue = "antibiotic",
            alpha = .5, palette = "muted",
            height = 6, data = data)

plt.title("AMR rate according to AMC")
plt.xlabel("AMC")
plt.ylabel("AMR rate (%)")

sns.despine(offset=10, trim=True)
plt.show()

#------------------------------------------------
# 2) Droite de régression même graphe
#-------------------------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8,5))

sns.regplot(x = "antibiotic_consumption", 
            y = "amr_rate", 
            data = data)

plt.title("AMR rate according to AMC")
plt.xlabel("AMC")
plt.ylabel("AMR rate (%)")

sns.despine(offset=10, trim=True)
plt.show()

#------------------------------------------------
# 4) Lineplot: evolution moyenne de l'AMR au cours du temps
#-------------------------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8,5))

sns.lineplot(x = "year", 
            y = "amr_rate",
            hue = "region",
            data = data)

plt.title("AMR rate time evolution")
plt.xlabel("Year")
plt.ylabel("AMR rate (%)")

sns.despine(offset=10, trim=True)
plt.show()

#------------------------------------------------
# 5) Barplot: conso antibitic en fonction de la région
#-------------------------------------------------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8,5))

sns.barplot(x = "region", 
            y = "antibiotic_consumption",
            hue = "region",
            data = data)

plt.xlabel("Region")
plt.ylabel("AMC")

sns.despine(offset=10, trim=True)
plt.show()

#------------------------------------------------
# 6) Facets scatterplot AMC VS AMR par type hopital
#-------------------------------------------------

sns.set_theme(style="whitegrid")

# Création des facets
grid = sns.FacetGrid(
    data,
    col="hospital_type",
    hue="hospital_type",
    col_wrap=3,
    height=4
)

# Tracé
grid.map(
    plt.scatter,
    "antibiotic_consumption",
    "amr_rate"
)

# Ajustements
grid.set_axis_labels(
    "Antibiotic consumption",
    "AMR rate"
)

grid.fig.tight_layout()

sns.despine(offset=10, trim=True)

plt.show()


