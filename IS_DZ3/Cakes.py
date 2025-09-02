import math

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
matplotlib.use("TkAgg")

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import re

pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 20)

pd.set_option('display.width', None)

data = pd.read_csv('datasets/cakes_train.csv')
# data.insert(loc=0, column='flour',value=np.arange(0, len(data), 1))

#1 - prvih i poslednjih 5 primeraka
print(data.head())
print("...")
print(data.tail())

print("\n\n")
print("--------------------------")
print("\n\n")

# 2 - koncizne informacije o sadrzaju tabele i statisticke informacije
# o svim atributima
print("INFORMACIJE O PRIMERCIMA:")
print(data.info())
print("\n")
print("KONTINUALNI PODACI:")
print(data.describe())
print("\n")
print("KATEGORICKI PODACI:")
print(data.describe(include=[object]))


print("\n\n")
print("--------------------------")
print("\n\n")

print("PODACI KOJI FALE:\n")
total = data.isnull().sum().sort_values(ascending=False)
perc1 = data.isnull().sum() / data.isnull().count() * 100
perc2 = (round(perc1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, perc2], axis=1, keys=['Total', '%'])
print(missing_data.head(7))


print("\n\n")
print("-------------PRIKAZ ODNOSA ULAZNIH I IZLAZNOG ATRIBUTA-------------")
print("\n\n")

#
data["eggs"] = data["eggs"] * 63
features = [c for c in data.columns if c not in ["type", "y", "y_jitter"]]

sb.set(style="whitegrid")
n = len(features)
cols = 2
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows), squeeze=False)
fig.subplots_adjust(hspace=0.7, top=0.9)
for i, feat in enumerate(features):
    ax = axes[i // cols][i % cols]

    sb.scatterplot(
        data=data,
        x=feat,
        y="type",
        hue="type",
        alpha=0.7,
        s=35,
        palette="Set2",
        ax=ax
    )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["cupcake", "muffin"])
    ax.set_ylabel("type")
    ax.set_xlabel(feat)
    ax.set_title(f"Zavisnost tipa kolača od '{feat}'")
    ax.legend(title="type", loc="best")


plt.show()

sb.set_style(style='whitegrid')
# sb.boxplot()
fig, axes = plt.subplots(1, len(features), figsize=(15, 3))
fig.set_size_inches(w=45.0, h=5.0)
fig.subplots_adjust(left=0.035, right=0.98, bottom=0.1, wspace=0.3)
for i,a in enumerate(features):
    sb.histplot(data=data, x=a, element="step", hue="type", stat="count", common_norm=False,
                palette="Set2", ax=axes[i], multiple="layer")


# plt.tight_layout()
plt.show()


print("\n\n")
print("------------KORELACIONA MATRICA--------------")
print("\n\n")

le = LabelEncoder()
data["type"] = le.fit_transform(data["type"])
f = plt.figure()
f.subplots_adjust(left=0.17, right=0.98, bottom=0.23, top=0.926)
plt.title("KORELACIONA MATRICA")
sb.heatmap(data.corr(), annot=True, fmt='.2f')
plt.show()

print("\n\n")
print("------------TRENIRANJE--------------")
print("\n\n")

#GRUPISANJE KOLONA
data.loc[(data['flour'] < 200), 'flour'] = 0
data.loc[(data['flour'] >= 200) & (data['flour'] < 380), 'flour'] = 1
data.loc[(data['flour'] >= 380), 'flour'] = 2

data.loc[(data['eggs'] < 100), 'eggs'] = 0
data.loc[(data['eggs'] >= 100) & (data['eggs'] < 150), 'eggs'] = 1
data.loc[(data['eggs'] >= 150) & (data['eggs'] < 300), 'eggs'] = 2
data.loc[(data['eggs'] >= 300), 'eggs'] = 3
# data.loc[(data['eggs'] < 100), 'eggs'] = 0
# data.loc[(data['eggs'] >= 100) & (data['eggs'] < 200), 'eggs'] = 1
# data.loc[(data['eggs'] >= 200), 'eggs'] = 2

# data.loc[(data['eggs'] < 200), 'eggs'] = 0
# data.loc[(data['eggs'] >= 200), 'eggs'] = 1


data.loc[(data['sugar'] < 200), 'sugar'] = 0
data.loc[(data['sugar'] >= 200) & (data['sugar'] <= 470), 'sugar'] = 1
data.loc[(data['sugar'] > 470) & (data['sugar'] < 1100), 'sugar'] = 2
data.loc[(data['sugar'] > 1100), 'sugar'] = 3
# data.loc[(data['sugar'] < 500), 'sugar'] = 0
# data.loc[(data['sugar'] >= 500), 'sugar'] = 1


data.loc[(data['milk'] <= 200), 'milk'] = 0
data.loc[(data['milk'] > 200) & (data['milk'] <= 400), 'milk'] = 1
data.loc[(data['milk'] > 400), 'milk'] = 2

# data.loc[(data['milk'] <= 112), 'milk'] = 0
# data.loc[(data['milk'] > 112) & (data['milk'] <= 200), 'milk'] = 1
# data.loc[(data['milk'] > 200) & (data['milk'] <= 400), 'milk'] = 2
# data.loc[(data['milk'] > 400), 'milk'] = 3

data.loc[(data['butter'] <= 100), 'butter'] = 0
data.loc[(data['butter'] > 100) & (data['butter'] < 180), 'butter'] = 1
data.loc[(data['butter'] >= 180), 'butter'] = 2

data.loc[(data['baking_powder'] <= 6), 'baking_powder'] = 0
data.loc[(data['baking_powder'] > 6) & (data['baking_powder'] <= 10), 'baking_powder'] = 1
data.loc[(data['baking_powder'] > 10), 'baking_powder'] = 2

# STAVLJAJ UZE KATEGORIJE

# print(data.head(10))

dtc = DecisionTreeClassifier(criterion='entropy')
X = data.drop(columns=['type']) # na osnovu cega se vrsi klasifikacija
y = data['type'] # labele - tacni odgovori

# formiranje train i test skupova
# x_train - skup za treniranje u odnosu na y_train
# x_test - skup za testiranje u odnosu na y_test

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=234, shuffle=True)

dtc.fit(x_train, y_train) # treniranje

predictions = dtc.predict(x_test)

prediction_series = pd.Series(data=predictions, name='PREDICTED', index=x_test.index)
resulting_data = pd.concat([x_test, y_test, prediction_series], axis=1)
print("\n")
print("----PREDICTIONS----")
print(resulting_data.head(10))
print(".....")
print(resulting_data.tail(10))
print("\n")
print(f"MODEL SCORE: {dtc.score(x_test, y_test):0.3f}")


# CrossValidation predstavlja tehniku validacije modela,
# koja radi po sledecem principu:
# podaci se podele na cv podskupova podataka, pri cemu se (cv-1) skupova
# koristi za treniranje, a 1 skup podataka za validaciju.
# Postupak se ponavlja cv puta tako da svaki skup po jednom ucestvuje
# kao skup za testiranje, kao takav.
cv = cross_validate(dtc, X, y, cv = 6)
print("\n")
# print(sorted(cv['test_score']))
print(f"CV SCORE: {cv['test_score'].mean():0.3f}")


#VIZUELIZACIJA STABLA ODLUCIVANJA

fig, ax = plt.subplots(1,1, figsize=(8,3), dpi = 200)
tree.plot_tree(decision_tree=dtc, max_depth=3, feature_names=X.columns, class_names=['Cupcake', 'Muffin'], fontsize=3, filled=True)
plt.show()

plt.figure()
feat = dict(zip(X.columns, dtc.feature_importances_)) # key: attr; val: importance
items = sorted(feat.items(), key = lambda item: item[1], reverse=True) #descending by value
keys, values = zip(*items) # unpacking a dictionary

plt.bar(x=range(len(keys)), height=values, align='center')
plt.xticks(ticks=range(len(keys)), labels=keys, rotation=90)
plt.title("ZNACAJ ATRIBUTA")
plt.show()

#--------------------------------------

data["eggs"] = data["eggs"] // 63
