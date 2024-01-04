import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sklearn.preprocessing as pp
import sklearn.metrics as skmet
from sklearn import cluster
import matplotlib.cm as cm
from scipy.optimize import curve_fit

Inf = pd.read_csv("Inflation_Data.csv")
print(Inf.describe())
# The plan is to use 2000 and 2020 for clustering. Countries with one NaN are 
Inf = Inf[(Inf["2000"].notna()) & (Inf["2020"].notna())]
Inf = Inf.reset_index(drop=True)
# extract 2000
growth = Inf[["Country Name", "2000"]].copy()
# and calculate the growth over 40 years
growth["Growth"] = 100.0/20.0 * (Inf["2020"]-Inf["2000"]) / Inf["2000"]
print(growth.describe())
print()
print(growth.dtypes)

plt.figure(figsize=(8, 8))
plt.scatter(growth["2000"], growth["Growth"])
plt.xlabel("Inflation,Consumer Prices,2000")
plt.ylabel("Growth per year [%]")
plt.show()

# create a scaler object
scaler = pp.RobustScaler()
# and set up the scaler
# extract the columns for clustering
df_ex = growth[["2000", "Growth"]]
scaler.fit(df_ex)
# apply the scaling
norm = scaler.transform(df_ex)
plt.figure(figsize=(8, 8))
plt.scatter(norm[:, 0], norm[:, 1])
plt.xlabel("Inflation,Consumer Prices,2000")
plt.ylabel("Growth per year [%]")
plt.show()

def one_silhoutte(xy, n):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy) # fit done on x,y pairs
    labels = kmeans.labels_
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))
    return score

#calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")

# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(norm) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(growth["2000"], growth["Growth"], 10, labels, marker="o", cmap=cm.rainbow)
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("Inflation,Consumer Prices,2000")
plt.ylabel("Growth/year [%]")
plt.show()

print(cen)

growth2 = growth[labels==0].copy()
print(growth2.describe())

df_ex = growth2[["2000", "Growth"]]
scaler.fit(df_ex)
# apply the scaling
norm = scaler.transform(df_ex)
plt.figure(figsize=(8, 8))
plt.scatter(norm[:, 0], norm[:, 1])
plt.xlabel("Inflation,Consumer Prices,2000")
plt.ylabel("Growth per year [%]")
plt.show()


# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(norm) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_
# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]
plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(growth2["2000"], growth2["Growth"], 10, labels, marker="o", cmap=cm.rainbow)
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
plt.xlabel("Inflation,Consumer Prices,2000")
plt.ylabel("Growth/year [%]")
plt.show()



#Load your data here
data = pd.read_csv('UK_Inf.csv')

# Example data columns: 'Year' and 'Inflation'
x = data['Year']
y = data['Inflation']

# Defining the cubic model function
def cubic_model(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Fitting the cubic model to the data
popt, _ = curve_fit(cubic_model, x, y)

# Creating a range of x values for plotting the fit and the forecast
x_fit = np.linspace(min(x), max(x), 1000)  # Dense range for a smooth curve
x_forecast = np.linspace(max(x), max(x) + 20, 21)  # Next 20 years

# Generating the fitted curve and the forecast
y_fit = cubic_model(x_fit, *popt)
y_forecast = cubic_model(x_forecast, *popt)
plt.figure(figsize=(8, 8))
# Plotting the historical data
plt.plot(x, y, label='Historical Data', color='blue')

# Plotting the fitted cubic model and the forecast
plt.plot(x_fit, y_fit, label='Fitted Model', color='orange')
plt.plot(x_forecast, y_forecast, label='Forecast', color='orange', linestyle='--')

# Labeling the axes and the plot
plt.xlabel('Year')
plt.ylabel('Inflation')
plt.title('UK Inflation Forecast for the Next 20 Years',fontsize = '13')
plt.legend()

# Display the plot
plt.show()
