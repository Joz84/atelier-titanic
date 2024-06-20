
# Atelier Titanic
## Support de cours
<a href="https://docs.google.com/presentation/d/1-TguitlJAfuk4JfkYnf4p13JBUJCbcUd8m0q4-C6NNE/edit?usp=sharing" target="_blank">Lien vers le support</a>

## Recupération des bibliothèques et du Jeu de données

### Recupération des bibliothèques
```
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, layers
```

### Recupération  du Jeu de données
```
titanic = sns.load_dataset("titanic")
```

## Netoyage et Foramatage des données

### Sélection des colonnes intéressantes
```
titanic = titanic.dropna(subset=["survived", "sex", "pclass", "age", "fare"])
titanic = titanic.reset_index()
titanic
```

### Convertir une colonne "texte" en colonne "Chiffre"
```
titanic["sex_num"] = titanic["sex"].map({"male": 0, "female": 1})
```

## Découpage du tableau pour le faire passer dans le modèle

### Sélection des colonnes à garder pour faire la prédiction
```
X = titanic[["sex_num", "pclass", "age", "fare"]]
X
```

### Quelle serait la ligne qui me correspond
```
my_X = pd.DataFrame({"sex_num": [0], "pclass": [1],	"age": [39], "fare": [50.0]})
my_X
```

### La colonne à prédire
```
y = titanic["survived"]
y
```

### Ma chance de survie en ne connaissant aucune infos sur moi
```
values = pd.Series(y).value_counts()
print(values)
values.max() / values.sum()
```

### Garder une partie des données pour tester le resultat
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=3)
```

### Mettre toutes les données sur un même pied d'égalité (standardisé)
```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
```

## Création du modèle
```
model = Sequential()
model.add(layers.Dense(10, activation='relu', input_dim=4)) 
model.add(layers.Dense(5, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
```

## Entrainement du modèle
```
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = 'accuracy')
model.fit(X_train, y_train, batch_size=16, epochs=20)
```

## Evaluer la fiabilité du modèle
```
model.evaluate(scaler.transform(X_test), y_test)
```

## La prédiction est une probabilité
```
model.predict(scaler.transform(X_test))
```

## Est ce que j'aurais survécu?
```
model.predict(scaler.transform(my_X))
```

### Ressources:
- <a src="https://scikit-learn.org/">https://scikit-learn.org/</a>
- <a src="https://leekwars.com/">https://leekwars.com/</a>
- <a src="https://www.kaggle.com/datasets">https://www.kaggle.com/datasets</a>
- <a src="https://www.data.gouv.fr/fr/">https://www.data.gouv.fr/fr/</a>
- <a src="https://youtu.be/L_4BPjLBF4E?si=suqX8vQe00pNy0qO">https://youtu.be/L_4BPjLBF4E?si=suqX8vQe00pNy0qO</a>
