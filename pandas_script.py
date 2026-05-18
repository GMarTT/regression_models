import pandas as pd
import matplotlib.pyplot as plt

data = {
    "order_id": [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
    "customer": ["Alice", "Bob", "Alice", "David", "Emma", "Bob", "Frank", "Emma"],
    "product": ["Laptop", "Mouse", "Keyboard", "Laptop", "Mouse", "Laptop", "Keyboard", "Mouse"],
    "category": ["PC", "Accessories", "Accessories", "PC", "Accessories", "PC", "Accessories", "Accessories"],
    "price": [1200, 25, 70, 1100, 20, 1300, 80, 30],
    "quantity": [1, 2, 1, 1, 3, 1, 2, 1],
    "date": pd.to_datetime([
        "2026-01-01", "2026-01-02", "2026-01-03", "2026-01-03",
        "2026-01-04", "2026-01-05", "2026-01-06", "2026-01-06"
    ])
}

df = pd.DataFrame(data)
df

#-----------------------------------------
# exploration de base
#-----------------------------------------

# 1) Afficher les 5 premières lignes
df.head(5)

# 2) Donner les dimensions du dataset
df.shape
# 8 lignes
# 7 colonnes

# 3) Types des variables
df.dtypes

#-------------------------------------------
# création de nouvelles variables
#-------------------------------------------

# 1) total = price * quantity
df["total"] = df["price"] * df["quantity"]
df

# 2) Quel est le produit le plius vendu en quantité ?
df.groupby("product")["quantity"].sum()
# 6 mouses vendues

# 3) Chiffre d'affaire total 
df["total"].sum() # 3970 euros

# 4) Client qui a dépensé le plus 
df.groupby("customer")["total"].sum()
# C'est Bob avec 1 350 euros dépensés

# 5) Quel client a dépensé le plus en moyenne par commande ?
df.groupby("customer")["total"].mean()
# David avec en moyenne 1 100 euros par commande

# 6) Quel jour a eu le plus grand chiffre d'affaire ?
df.groupby("date")["total"].sum()
# le 5 janvier 2026

#-------------------------------------------------------------------------------
# niveau 2: entretien data analyst
#-------------------------------------------------------------------------------

# 1) créer un dataset en format wide client * category avec le total
df.pivot_table(
    index="customer",
    columns="category",
    values="total",
    aggfunc="sum",
    fill_value=0
)

# 2) Commande accesories avec plus de 2
df.query("category == 'Accessories' and quantity >= 2")

# 3) Commandes faites après le 2026-01-05 et total > 100
df.query("date > '2026-01-05' and total > 100")

# 4) Chiffre d'affaire par jour
dg = df.groupby("date")["total"].sum()

plt.plot(dg.index, dg.values)
plt.xlabel("Date")
plt.ylabel("CA")
plt.title("CA journalier")
plt.show()

# moyenne mobile sur 3 jours
dg = pd.Series(df.groupby("date")["total"].sum())
dg.rolling(3).mean()
# ça dépend de l'échelle, mais le CA est plutôt volatille

# 5) évolution des totaux par date
df.pivot_table(
    index="date",
    columns="category",
    values="total",
    aggfunc="sum",
    fill_value=0
)
# la catégorie PC domine sur la durée

# 6) les n clients qui ont le plus dépenser
def top_n_clients(n, df):
  dg = df.groupby("customer")["total"].sum().sort_values(ascending = False)
  return(dg.head(n))
   
top_n_clients(3, df)
