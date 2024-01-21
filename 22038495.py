import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit

"""@author: Muhammad Shoaib Manzoor 22038495"""

# Initializing dataset and scaler for use.
scaler = StandardScaler()
dataframe = pd.read_excel('./WaterResources.xlsx')

# Converting wide dataset into long for better maneuverability.
df_melted = pd.melt(dataframe, id_vars=[
                    'Country Code', 'Country Name',
                    'Series Name', 'Series Code'],
                    var_name='Year', value_name='Value')
top_level_df = df_melted.drop(['Country Name', 'Series Name'], axis=1)
top_level_df = top_level_df.dropna()
top_level_df = top_level_df.pivot(index=['Country Code', 'Year'],
                                  columns='Series Code', values='Value')
top_level_df = top_level_df.groupby('Country Code').mean()

# Dropping columns with higher counts of NaN values.
top_level_df.drop(['AG.LND.IRIG.AG.ZS', 'SH.STA.HYGN.ZS'],
                  axis=1, inplace=True)

# Dropping rows with NaN values.
top_level_df.dropna(inplace=True)

# Creating a subset of Precipitation & Water Stress Level
stressXprecip = top_level_df[['AG.LND.PRCP.MM', 'ER.H2O.FWST.ZS']]

# Scaling selected features for clustering.
feats_scaled = scaler.fit_transform(stressXprecip)

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, n_init=10)

# Merging assigned cluster values into dataset.
top_level_df['sXp_cluster'] = kmeans.fit_predict(feats_scaled)

sns.set(style='darkgrid')

fig, ax = plt.subplots(figsize=(14, 8))

# First clustering plot with Water Stress against Precipitation.
ax.scatter(
    x=top_level_df['ER.H2O.FWST.ZS'],
    y=top_level_df['AG.LND.PRCP.MM'],
    c=top_level_df['sXp_cluster'],
    cmap='viridis',
    vmin=-1,
    vmax=3
)
ax.title('Level of water stress Vs. Precipitation', fontsize=16)
ax.xlabel('Level of water stress', fontsize=14)
ax.ylabel('Precipitation', fontsize=14)
ax.xlim(-10, 200)

# Creating a subset of values narrowed down from cluster analysis.
subset = top_level_df[(top_level_df['ER.H2O.FWST.ZS'] > 75)
                      & (top_level_df['AG.LND.PRCP.MM'] < 1000)]
trend_set = pd.merge(df_melted, subset, on='Country Code', how='inner')
trend_set = trend_set[['Country Name', 'Country Code',
                       'Series Code', 'Series Name', 'Year', 'Value']]

# Creating subplots of countries with trends of water stress.
water_stress = trend_set[trend_set['Series Code'] == 'ER.H2O.FWST.ZS']
water_stress = water_stress.pivot(
    index='Country Code', columns='Year', values='Value')

co2_emissions = trend_set[trend_set['Series Code'] == 'EN.ATM.CO2E.PC']
co2_emissions = co2_emissions.pivot(
    index='Country Code', columns='Year', values='Value')


sub_watstr = water_stress[9:12]

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(sub_watstr.T, marker='o', label=sub_watstr.index)
plt.title('Trend of water Stress over years', fontsize=16)
plt.ylabel('Level of water stress', fontsize=14)
plt.xticks(list(map(int, water_stress.columns[::2])))
plt.legend()


pak_sub_watstr = water_stress.loc['PAK']

fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(pak_sub_watstr.T, marker='o', label='Pakistan')
plt.title('Trend of water Stress over years for Pakistan', fontsize=16)
plt.ylabel('Level of water stress', fontsize=14)
plt.xticks(list(map(int, water_stress.columns[::2])))
plt.legend()

pak_sub_co2e = co2_emissions.loc['PAK']

# Designing a curve fit model to curve_fit.

# Using quadratic ploynomial for generating curve.


def quad_func(x, a, b, c):
    return a * x**2 + b * x + c


years = water_stress.columns
prediction_years = np.arange(2001, 2031)

parameters, covar_actual = curve_fit(
    quad_func, years, water_stress.loc['PAK'].values)

a_fit, b_fit, c_fit = parameters

predicted_values = quad_func(prediction_years, a_fit, b_fit, c_fit)

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(prediction_years, predicted_values,
        label='Predicted', color='green', linestyle='dashed')
ax.plot(years, water_stress.loc['PAK'].values,
        marker='o', label='Current', color='maroon')

plt.title('Prediction Curve for Water Stress Level', fontsize=16)
plt.ylabel('Water Stress Level', fontsize=14)
plt.xticks(list(map(int, water_stress.columns[::2])))
plt.legend()


parameters, covar_actual = curve_fit(
    quad_func, years, co2_emissions.loc['PAK'].values)

a_fit, b_fit, c_fit = parameters

predicted_values = quad_func(prediction_years, a_fit, b_fit, c_fit)

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(prediction_years, predicted_values,
        label='Predicted', color='green', linestyle='dashed')
ax.plot(years, co2_emissions.loc['PAK'].values,
        marker='o', label='Current', color='maroon')
plt.title('Prediction Curve for CO2 Emissions', fontsize=16)
plt.ylabel('CO2 Emissions', fontsize=14)
plt.xticks(list(map(int, water_stress.columns[::2])))
plt.legend()

plt.show()
