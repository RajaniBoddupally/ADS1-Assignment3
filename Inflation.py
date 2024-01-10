import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from sklearn import cluster
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


def read_file(file_path):
    """Reads a CSV file and returns a DataFrame."""
    data = pd.read_csv(file_path)
    return data


def calculate_silhouette_score(xy, n_clusters):
    """Calculates the silhouette score for a given number of clusters."""
    kmeans = cluster.KMeans(n_clusters=n_clusters, n_init=20)
    kmeans.fit(xy)
    labels = kmeans.labels_
    score = silhouette_score(xy, labels)
    return score


def polynomial(x, a, b, c, d):
    """Polynomial model function."""
    return a * x**3 + b * x**2 + c * x + d


def calculate_error(x, func, params, covariance):
    """Calculates error propagation."""
    var = np.zeros_like(x)
    for i in range(len(params)):
        derivative1 = calculate_derivative(x, func, params, i)
        for j in range(len(params)):
            derivative2 = calculate_derivative(x, func, params, j)
            var += derivative1 * derivative2 * covariance[i, j]
    return np.sqrt(var)


def calculate_derivative(x, func, params, index):
    """Calculates derivative for error propagation."""
    scale = 1e-6
    delta = np.zeros_like(params)
    delta[index] = scale * abs(params[index])
    param_plus = params + delta
    param_minus = params - delta
    diff = 0.5 * (func(x, *param_plus) - func(x, *param_minus))
    return diff / (scale * abs(params[index]))


# Main analysis starts here
inflation_data = read_file("Inflation_Data.csv")
if inflation_data is not None:
    print(inflation_data.describe())

inflation_data = inflation_data[(inflation_data["2000"].notna()) & 
                                 (inflation_data["2020"].notna())]
inflation_data.reset_index(drop=True, inplace=True)

growth = inflation_data[["Country Name", "2000"]].copy()
growth["Growth"] = 100.0 / 20.0 * (inflation_data["2020"] - 
                    inflation_data["2000"]) / inflation_data["2000"]

print(growth.describe())
print()
print(growth.dtypes)

plt.figure(figsize=(8, 8))
plt.scatter(growth["2000"], growth["Growth"])
plt.xlabel("Inflation, Consumer Prices, 2000")
plt.ylabel("Growth per year [%]")
plt.show()

# Data normalization
scaler = RobustScaler()
df_for_clustering = growth[["2000", "Growth"]]
scaler.fit(df_for_clustering)
normalized_data = scaler.transform(df_for_clustering)

plt.figure(figsize=(8, 8))
plt.scatter(normalized_data[:, 0], normalized_data[:, 1])
plt.xlabel("Normalized Inflation, Consumer Prices, 2000")
plt.ylabel("Normalized Growth per year [%]")
plt.show()

# Silhouette analysis
for cluster_count in range(2, 11):
    score = calculate_silhouette_score(normalized_data, cluster_count)
    score_message = f"The silhouette score for {cluster_count:3d} clusters is"
    print(f"{score_message} {score:7.4f}")


# Clustering with KMeans
kmeans = cluster.KMeans(n_clusters=3, n_init=20)
kmeans.fit(normalized_data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
original_centers = scaler.inverse_transform(centers)

plt.figure(figsize=(8.0, 8.0))
plt.scatter(growth["2000"], growth["Growth"], c=labels, 
            s=10, marker="o", cmap=cm.rainbow)
plt.scatter(original_centers[:, 0], original_centers[:, 1],
            s=45, c="k", marker="d")
plt.xlabel("Inflation, Consumer Prices, 2000")
plt.ylabel("Growth/year [%]")
plt.show()

print(original_centers)

growth_subset = growth[labels == 0].copy()
print(growth_subset.describe())

df_for_second_clustering = growth_subset[["2000", "Growth"]]
scaler.fit(df_for_second_clustering)
normalized_subset_data = scaler.transform(df_for_second_clustering)

plt.figure(figsize=(8, 8))
plt.scatter(normalized_subset_data[:, 0], normalized_subset_data[:, 1])
plt.xlabel("Inflation, Consumer Prices, 2000")
plt.ylabel("Growth per year [%]")
plt.show()

# Second clustering within subset
kmeans_subset = cluster.KMeans(n_clusters=3, n_init=20)
kmeans_subset.fit(normalized_subset_data)
subset_labels = kmeans_subset.labels_
subset_centers = kmeans_subset.cluster_centers_
original_subset_centers = scaler.inverse_transform(subset_centers)

plt.figure(figsize=(8.0, 8.0))
plt.scatter(growth_subset["2000"], growth_subset["Growth"], 
            c=subset_labels, s=10, marker="o", cmap=cm.rainbow)
plt.scatter(original_subset_centers[:, 0], original_subset_centers[:, 1],
            s=45, c="k", marker="d")
plt.xlabel("Inflation, Consumer Prices, 2000")
plt.ylabel("Growth/year [%]")
plt.show()

# Load and transpose UK inflation data
uk_inflation_data = read_file('UK_Inf.csv')
uk_inf_data_trsp = uk_inflation_data.T

# Cleaning the transposed data
uk_inf_data_trsp.columns = ['Inflation']
uk_inf_data_trsp = uk_inf_data_trsp.drop('Year')
uk_inf_data_trsp.reset_index(inplace=True)
uk_inf_data_trsp.rename(columns={'index': 'Year'}, inplace=True)
uk_inf_data_trsp['Year'] = uk_inf_data_trsp['Year'].astype(int)
uk_inf_data_trsp['Inflation'] = uk_inf_data_trsp['Inflation'].astype(float)

# Extracting x and y values for modeling
x_values = uk_inf_data_trsp['Year'].values.astype(float)
y_values = uk_inf_data_trsp['Inflation'].values.astype(float)

# Fitting the polynomial model to the data
popt, pcov = curve_fit(polynomial, x_values, y_values)

# Calculate error ranges for existing data
y_error = calculate_error(x_values, polynomial, popt, pcov)

# Generate future years and predict values
x_future = np.arange(max(x_values) + 1, 2031)
y_future = polynomial(x_future, *popt)

# Calculate error ranges for future predictions
y_future_error = calculate_error(x_future, polynomial, popt, pcov)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, 'b-', label='Historical Data')
plt.plot(x_values, polynomial(x_values, *popt), 'r-', 
         label='Fitted Polynomial Model')
plt.fill_between(x_values, polynomial(x_values, *popt) - 
    y_error, polynomial(x_values, *popt) + y_error, color='orange', 
    alpha=0.5, label='Confidence Interval for Historical Data')
plt.plot(x_future, y_future, 'g--', label='Future Predictions')
plt.fill_between(x_future, y_future - y_future_error, y_future +
                 y_future_error, color='lightgreen', 
                 alpha=0.5, label='Confidence Interval for Future Predictions')
plt.xlabel('Year')
plt.ylabel('Inflation')
plt.title('UK Inflation Forecast with Confidence Intervals')
plt.legend()
plt.show()
