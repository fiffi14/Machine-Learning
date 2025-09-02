import matplotlib
import pandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from matplotlib import cm  # za bojenje fje greske
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

matplotlib.use("TkAgg")

from sklearn.linear_model import LinearRegression

def show_corr_matrix(data):
    f = plt.figure("KORELACIONA MATRICA")
    f.subplots_adjust(left=0.17, right=0.98, bottom=0.23, top=0.926)
    plt.title("KORELACIONA MATRICA")
    sb.heatmap(data.corr(), annot=True, square=True, fmt='.2f')
    plt.show()


def show_pairplot(data):

    sb.pairplot(data)
    plt.show()


def show_categorical_data(data_):
    plot_data = data.groupby(data_)["Price"].mean()
    print(plot_data)
    plt.figure("Zavisnost Price od " + data_)
    plt.title("Zavisnost Price od " + data_)
    plt.xlabel(data_)
    plt.ylabel("Price")
    plot_data.plot.bar()
    plt.tight_layout()
    plt.show()


def show_continuous_data(data_):
    data_in = data[[data_]]
    data_out = data['Price']

    plt.figure('Zavisnost Price od ulaza ' + data_)
    plt.title('Zavisnost Price od ulaza ' + data_)
    plt.xlabel(data_)
    plt.ylabel('Price')
    plt.scatter(data_in, data_out, s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2 ,label=data_)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
# sb.regplot(x="Area", y="Price", data=data, ax=axes[0], line_kws={"color":"red"})
# sb.regplot(x="Year_built", y="Price", data=data, ax=axes[1], line_kws={"color":"red"})
# sb.regplot(x="Bath_no", y="Price", data=data, ax=axes[2], line_kws={"color":"red"})
#
# plt.show()


data = pd.read_csv('datasets/house_prices_train.csv')

# DATA INFO

print(data.head())
print("...")
print(data.tail())

print("\n")
print(data.info())
print(data.describe())

# show_corr_matrix(data)

# CLEANSING

# Check if data is null
# print("***********************")
# print(data.loc[data['Bedroom_no'].isnull()])
#
# print(type(data[COLUMN].mean())) => CHECK THE TYPE

# data[COLUMN] = data[COLUMN].fillna(np.around(data[COLUMN].mean(), decimals=1))
# data[COLUMN] = data[COLUMN].fillna(data[COLUMN].mode()[0])

# FEATURE ENGINEERING

X = data.loc[:, ['Year_built', 'Area', 'Bath_no', 'Bedroom_no']]
y = data['Price']


X['Year_built'] = StandardScaler().fit_transform(X[['Year_built']])
X['Area'] = MinMaxScaler().fit_transform(np.log1p(X[['Area']]))

y = np.log1p(y)

# X['Area'] = (X['Area'] - X['Area'].mean()) / X['Area'].std()
# X['Year_built'] = (X['Year_built'] - X['Year_built'].mean()) / X['Year_built'].std()
# X['Bath_no'] = (X['Bath_no'] - X['Bath_no'].min()) / (X['Bath_no'].max() - X['Bath_no'].min())

# X['Area'] = (X['Area'] - X['Area'].min()) / (X['Area'].max() - X['Area'].min())
# X['Year_built'] = (X['Year_built'] - X['Year_built'].min()) / (X['Year_built'].max() - X['Year_built'].min())
# X['Bath_no'] = (X['Bath_no'] - X['Bath_no'].min()) / (X['Bath_no'].max() - X['Bath_no'].min())
# X['Bedroom_no'] = (X['Bedroom_no'] - X['Bedroom_no'].min()) / (X['Bedroom_no'].max() - X['Bedroom_no'].min())

# X['Bath_no'] = X['Bath_no'].map(lambda x: float(x))
# X['Bedroom_no'] = X['Bedroom_no'].map(lambda x: float(x))


# y = (y - y.min()) / (y.max() - y.min())
# y = y / 1000

print("\n SKALIRANE VREDNOSTI\n")
print(X)
print(y)

show_corr_matrix(X)

show_pairplot(data)
show_continuous_data("Year_built")
show_continuous_data("Area")
show_categorical_data("Bath_no")
show_categorical_data("Bedroom_no")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
sb.regplot(x="Area", y="Price", data=data, ax=axes[0], line_kws={"color":"red"})
sb.regplot(x="Year_built", y="Price", data=data, ax=axes[1], line_kws={"color":"red"})
sb.regplot(x="Bath_no", y="Price", data=data, ax=axes[2], line_kws={"color":"red"})

plt.show()

# *************************************

X['intercept'] = 1

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=112, shuffle=False)

coeffs = np.random.rand(X_train.shape[1])
mseList = []
alpha = 0.15
iters = 10000 #3000

for i in range(iters):
    prediction = X_train.dot(coeffs) # h(xi)

    error = prediction - y_train # h(xi) - yi


    gradient = X_train.T.dot(error) / X_train.shape[0]
    # print(gradient)
    coeffs = coeffs - alpha * gradient
    mse = np.mean(error ** 2)
    mseList.append(mse)

X_test['intercept'] = 1
prediction = X_test.dot(coeffs) # x1w1 + x2w2

print("\n---------------------\nCOEFFICIENTS OF MY MODEL: \n"+ str(coeffs))
print("Final MSE: ", mseList[-1])
print("Score: " + str(r2_score(y_test, prediction)))

plt.figure('MS Error')
plt.plot(np.arange(0, len(mseList[:100]), 1), mseList[:100])
plt.xlabel('Iteration', fontsize=13)
plt.ylabel('MS error value', fontsize=13)
plt.xticks(np.arange(0, len(mseList[:100]), 2))
plt.title('Mean-square error function')
plt.tight_layout()
plt.legend(['MS Error'])
plt.show()


# Skaliramo vrednosti labela na skalu povrsina nekretnina.
# Bitno je da su vrednosti na priblizno slicnoj skali.
# U suprotnom, korak ucenja bi bio prilagodjen samo
# koeficijentima atributa sa vrednostima na istoj skali.
# Time ubrzavamo algoritam gradijentnog spusta.

#
# # ----------------------------------------------------
#
#
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

pred_lib = lr_model.predict(X_test)
pred_ser = pd.Series(data = pred_lib, name="Predicted", index=X_test.index)
res_df = pd.concat([X_test, y_test, pred_ser], axis=1)
print("\nDATA WITH PREDICTIONS - built-in")
print(res_df.head())
print("...")
print(res_df.tail())

np.set_printoptions(formatter={'float': lambda x: "{0:0.10f}".format(x)})

print("BUILT-IN MODEL")
print(f"\n\nCOEFFICIENTS: \n {lr_model.coef_}")
print(f"Mean Squared Error: " + str(mean_squared_error(y_test, pred_ser)))
print("Model score: " + str(lr_model.score(X_test, y_test)))


print("\n\nSKUP PODATAKA ZA TRENIRANJE\n")
print(X_train)
print("\n\nSKUP PODATAKA ZA VALIDACIJU\n")
print(X_test)

