# %% Imports & Styling
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

# Global styling
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.titlecolor'] = 'blue'
mpl.rcParams['axes.labelcolor'] = 'darkred'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.4

sns.set_theme(style="whitegrid")


# %% Stage 1: Load and clean dataset
url = "https://github.com/JMunetsi/Colon_Cancer_Dashboard/raw/refs/heads/main/colorectal_cancer_dataset.csv"
df = pd.read_csv(url)


df.columns = df.columns.str.strip().str.lower()


for col in df.select_dtypes(include="object"):
    df[col] = df[col].astype(str).str.strip().str.lower()

num_cols = [
    'age', 'tumor_size_mm', 'healthcare_costs',
    'incidence_rate_per_100k', 'mortality_rate_per_100k'
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df_sample = df.sample(min(len(df), 1000), random_state=42)

# Ordered categories
df['cancer_stage'] = pd.Categorical(
    df['cancer_stage'], ['localized', 'regional', 'metastatic'], ordered=True
)

df['obesity_bmi'] = pd.Categorical(
    df['obesity_bmi'], ['normal', 'overweight', 'obese'], ordered=True
)

# Summary Table

from prettytable import PrettyTable


num_cols = ['age', 'tumor_size_mm', 'healthcare_costs',
            'incidence_rate_per_100k', 'mortality_rate_per_100k']

table = PrettyTable()
table.field_names = ["Feature", "Mean", "Median", "Std Dev", "Min", "Max"]

for col in num_cols:
    table.add_row([
        col,
        round(df[col].mean(), 2),
        round(df[col].median(), 2),
        round(df[col].std(), 2),
        round(df[col].min(), 2),
        round(df[col].max(), 2)
    ])

print("\nNUMERICAL SUMMARY TABLE")
print(table)



# %% Stage 2: Plots

# Line plot
plt.figure(figsize=(10,5))

rolling_series = df['incidence_rate_per_100k'].rolling(window=500).mean()

plt.plot(rolling_series, linewidth=2, color='blue')
plt.title("Incidence Rate ")
plt.xlabel("Index")
plt.ylabel("Incidence Rate per 100k")
plt.grid(True)
plt.show()

# Grouped bar plot
plt.figure(figsize=(8,4))
sns.countplot(data=df, x="cancer_stage", hue="gender", palette="Set2")
plt.title("Grouped Bar – Cancer Stage by Gender")
plt.xlabel("Cancer Stage")
plt.ylabel("Count")
plt.show()

# Stacked bar plot
ct = pd.crosstab(df['cancer_stage'], df['gender'])
ct.plot(kind='bar', stacked=True, figsize=(8,8), colormap="Set3")
plt.title("Stacked Bar Plot – Cancer Stage by Gender")
plt.xlabel("Cancer Stage")
plt.ylabel("Count")
plt.grid()
plt.show()

# Count plot
plt.figure(figsize=(8,8))
sns.countplot(data=df, x="treatment_type", palette="Set2")
plt.title("Treatment Type Count Plot")
plt.xlabel("Treatment Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# Pie chart
plt.figure(figsize=(6,6))
df['gender'].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("Set2"))
plt.title("Gender Distribution Pie Chart")
plt.ylabel("")
plt.show()

# Dist plot
sns.displot(df['age'], kde=True, height=4, aspect=1.5)
plt.title("Age Distribution Plot")
plt.xlabel("Age")
plt.show()

#Pair plot
sns.pairplot(df_sample[['age', 'tumor_size_mm', 'healthcare_costs']], diag_kind="kde")
plt.show()

# Heatmap
corr = df.select_dtypes(include='number').corr()
plt.figure(figsize=(16,12))
sns.heatmap(corr, annot=True, cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()

# Histogram + KDE
plt.figure(figsize=(8,4))
sns.histplot(df['age'], kde=True, color="purple")
plt.title("Histogram with KDE – Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# QQ plot
plt.figure(figsize=(6,4))
stats.probplot(df['age'].dropna(), plot=plt)
plt.title("QQ Plot – Age")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")
plt.grid()
plt.show()

# Filled KDE
plt.figure(figsize=(8,4))
sns.kdeplot(df['healthcare_costs'], fill=True, alpha=0.6, linewidth=2, color="green")
plt.title("Filled KDE – Healthcare Costs")
plt.xlabel("Healthcare Costs")
plt.show()

# Regression plot
plt.figure(figsize=(8,4))
sns.regplot(x='age', y='tumor_size_mm', data=df, scatter_kws={'alpha':0.5})
plt.title("Regression Plot – Age vs Tumor Size")
plt.xlabel("Age")
plt.ylabel("Tumor Size (mm)")
plt.show()

# Multivariate box
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['age', 'tumor_size_mm', 'healthcare_costs']])

scaled_df = pd.DataFrame(scaled, columns=['age', 'tumor_size_mm', 'healthcare_costs'])
plt.figure(figsize=(8,4))
scaled_df.plot(kind='box')
plt.title("Normalized Box Plot – Age, Tumor Size, Healthcare Costs")
plt.ylabel("Scaled Value")
plt.grid(True)
plt.show()

#Multivariate boxen
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['age', 'tumor_size_mm', 'healthcare_costs']])
scaled_df = pd.DataFrame(scaled, columns=['age', 'tumor_size_mm', 'healthcare_costs'])
plt.figure(figsize=(8,4))
sns.boxenplot(data=scaled_df, palette="Set3")
plt.title("Normalized Boxen Plot – Age, Tumor Size, Healthcare Costs")
plt.ylabel("Scaled Value")
plt.show()

# Area plot
plt.figure(figsize=(8,4))
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['age', 'tumor_size_mm']].head(200))
scaled_df = pd.DataFrame(scaled, columns=['age', 'tumor_size_mm'])
scaled_df.plot.area(alpha=0.6, color=['#1f77b4', '#ff7f0e'])
plt.title("Normalized Area Plot – Age & Tumor Size")
plt.xlabel("Index")
plt.ylabel("Scaled Value")
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Violin plot
plt.figure(figsize=(8,4))
sns.violinplot(data=df, x='gender', y='age', palette="Set2")
plt.title("Violin Plot – Age by Gender")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()

# Joint plot (KDE + scatter)
sns.jointplot(data=df_sample, x='age', y='tumor_size_mm', kind='kde', fill=True, cmap="magma")
plt.show()

# Rug plot
plt.figure(figsize=(8,3))
sns.kdeplot(df['age'], fill=True, alpha=0.4)
sns.rugplot(df['age'], height=0.05, color='black')
plt.title("KDE with Rug Overlay – Age")
plt.xlabel("Age")
plt.show()

# 3D scatter plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_sample['age'], df_sample['healthcare_costs'], df_sample['tumor_size_mm'], alpha=0.6)
ax.set_xlabel("Age")
ax.set_ylabel("Healthcare Costs")
ax.set_zlabel("Tumor Size (mm)")
ax.set_title("3D Scatter Plot")
plt.show()

# Contour plot
plt.figure(figsize=(8,4))
sns.kdeplot(data=df_sample, x="age", y="tumor_size_mm", fill=True, cmap="viridis")
plt.title("Contour Plot – Age vs Tumor Size")
plt.xlabel("Age")
plt.ylabel("Tumor Size (mm)")
plt.show()

# Cluster map

corr_safe = corr.replace([np.inf, -np.inf], np.nan).fillna(0)

cluster = sns.clustermap(
    corr_safe,
    cmap="coolwarm",
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    figsize=(8, 6),
    cbar_pos=(0.02, 0.8, 0.03, 0.18)
)

cluster.fig.suptitle("Cluster Map – Correlation Matrix", fontsize=14, color='blue')
plt.subplots_adjust(top=0.9)
plt.show()

# Hexbin
plt.figure(figsize=(8,4))
df.plot.hexbin(x='age', y='tumor_size_mm', gridsize=25, cmap="viridis")
plt.title("Hexbin Plot – Age vs Tumor Size")
plt.xlabel("Age")
plt.ylabel("Tumor Size (mm)")
plt.show()

# Strip plot
plt.figure(figsize=(8,4))
sns.stripplot(data=df, x='gender', y='age', jitter=True, palette="Set2")
plt.title("Strip Plot – Age by Gender")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()

# Swarm plot
plt.figure(figsize=(8,4))
sns.swarmplot(data=df_sample, x='gender', y='age', size=3, palette="Set2")
plt.title("Swarm Plot – Age by Gender")
plt.xlabel("Gender")
plt.ylabel("Age")
plt.show()


# %% Stage 3: Storytelling Subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(data=df, x="gender", palette="Set2", ax=axs[0, 0])
axs[0, 0].set_title("Gender Distribution")
axs[0, 0].set_xlabel("Gender")
axs[0, 0].set_ylabel("Count")

sns.countplot(data=df, x="cancer_stage", hue="gender", palette="Set3", ax=axs[0, 1])
axs[0, 1].set_title("Cancer Stage by Gender")
axs[0, 1].set_xlabel("Cancer Stage")
axs[0, 1].set_ylabel("Count")

counts = df['obesity_bmi'].value_counts()
axs[1, 0].pie(counts, labels=counts.index, autopct="%1.1f%%", colors=sns.color_palette("Set2"))
axs[1, 0].set_title("BMI Categories")

sns.countplot(data=df, x="urban_or_rural", hue="gender", palette="Set2", ax=axs[1, 1])
axs[1, 1].set_title("Urban vs Rural by Gender")
axs[1, 1].set_xlabel("Urban or Rural")
axs[1, 1].set_ylabel("Count")

plt.tight_layout()
plt.show()
