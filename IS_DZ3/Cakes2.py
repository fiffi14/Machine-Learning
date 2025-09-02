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


data["eggs"] = data["eggs"] * 63
cols = [c for c in data.columns if c!="type"]

data["total"] = data[cols].sum(axis=1)
for c in cols:
    data[f"{c}"] = round(data[c]/data["total"]*100, 3)

print(data.head(10))

sb.pairplot(data)
plt.show()


sb.lmplot(data,x="flour", y="sugar", hue="type", palette="Set1", fit_reg=False, scatter={"s":70})
sb.lmplot(data,x="sugar", y="milk", hue="type", palette="Set1", fit_reg=False, scatter={"s":70})
sb.lmplot(data,x="sugar", y="baking_powder", hue="type", palette="Set1", fit_reg=False, scatter={"s":70})
sb.lmplot(data,x="flour", y="baking_powder", hue="type", palette="Set1", fit_reg=False, scatter={"s":70})
sb.lmplot(data,x="milk", y="baking_powder", hue="type", palette="Set1", fit_reg=False, scatter={"s":70})
plt.show()

le = LabelEncoder()
data["type"] = le.fit_transform(data["type"])
f = plt.figure()
f.subplots_adjust(left=0.17, right=0.98, bottom=0.23, top=0.926)
plt.title("KORELACIONA MATRICA")
sb.heatmap(data.corr(), annot=True, fmt='.2f')
plt.show()

data["type"] = le.inverse_transform(data["type"])
cols = [c for c in cols if c not in ["total"]]
sb.set_style(style='whitegrid')
# sb.boxplot()
fig, axes = plt.subplots(1, len(cols), figsize=(15, 3))
fig.set_size_inches(w=45.0, h=5.0)
fig.subplots_adjust(left=0.035, right=0.98, bottom=0.1, wspace=0.3)
for i,a in enumerate(cols):
    sb.histplot(data=data, x=a, element="step", hue="type", stat="count", common_norm=False,
                palette="Set2", ax=axes[i], multiple="layer", bins=20)


# plt.tight_layout()
plt.show()

data.loc[(data['flour'] < 20), 'flour'] = 0
data.loc[(data['flour'] >= 20) & (data['flour'] < 27), 'flour'] = 1
data.loc[(data['flour'] >= 27), 'flour'] = 2

data.loc[(data['eggs'] < 10), 'eggs'] = 0
data.loc[(data['eggs'] >= 10) & (data['eggs'] < 15), 'eggs'] = 1
data.loc[(data['eggs'] >= 15), 'eggs'] = 2

data.loc[(data['sugar'] < 18), 'sugar'] = 0
data.loc[(data['sugar'] >= 18) & (data['sugar'] < 42), 'sugar'] = 1
data.loc[(data['sugar'] >= 42), 'sugar'] = 2

data.loc[(data['milk'] < 18), 'milk'] = 0
data.loc[(data['milk'] >= 18) & (data['milk'] < 31), 'milk'] = 1
data.loc[(data['milk'] >= 31), 'milk'] = 2

data.loc[(data['butter'] < 11), 'butter'] = 0
data.loc[(data['butter'] >= 11), 'butter'] = 1

data.loc[(data['baking_powder'] < 1), 'baking_powder'] = 0
data.loc[(data['baking_powder'] >= 1), 'baking_powder'] = 1

dtc = DecisionTreeClassifier(criterion='entropy')
X = data.drop(columns=['type', 'total']) # na osnovu cega se vrsi klasifikacija
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
print(f"CV SCORE: {cv['test_score'].mean():0.3f}\n")

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

print("SKUP PODATAKA ZA TRENIRANJE")
print(x_train)

print("\n")
print("SKUP PODATAKA ZA VALIDACIJU")
print(x_test)

