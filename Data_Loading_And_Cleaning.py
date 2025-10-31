#%%
import pandas as pd
from prettytable import PrettyTable

#%% Loading the Dataset

url = "https://github.com/JMunetsi/Colon_Cancer_Dashboard/raw/refs/heads/main/colorectal_cancer_dataset.csv"
df = pd.read_csv(url)

#%% Displaying the Dataset
print ("Five First Rows of the Dataset:")
print(df.head().to_string())

#%%
print("Dataset Shape: ", df.shape)

#%%
print("Dataset Columns: ", df.columns)

#%%
print("Dataset Information:")
print(df.info())

#%%
print("Descriptive Statistics:")
print(df.describe().round(2).to_string())

#%% Summary for Missing Values

summary = df.isna().sum()
missing_table = PrettyTable()
missing_table.field_names = ["Column", "Missing Values"]

for col, val in summary.items():
    missing_table.add_row([col, val])

print("\n Summary for Missing Values:")
print(missing_table.get_string(title="Overview for Missing Data "))

#%% Summary for Data types
dtype_table = PrettyTable()
dtype_table.field_names = ["Column", "Data Type"]

for col in df.columns:
    dtype_table.add_row([col, df[col].dtype])

print("\nData Types:")
print(dtype_table.get_string(title="Column Data Types"))


#%% Cleaning the Dataset
df.columns = df.columns.str.strip().str.lower()


#%% Standardizing categorical values

cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

#%% Ordinal Features to ordered features

stage_order = ['localized', 'regional', 'metastatic']
diet_order = ['low', 'moderate', 'high']
activity_order = ['low', 'moderate', 'high']
access_order = ['low', 'moderate', 'high']
economic_order = ['developing', 'developed']

df['cancer_stage'] = pd.Categorical(df['cancer_stage'], categories=stage_order, ordered=True)
df['diet_risk'] = pd.Categorical(df['diet_risk'], categories=diet_order, ordered=True)
df['physical_activity'] = pd.Categorical(df['physical_activity'], categories=activity_order, ordered=True)
df['healthcare_access'] = pd.Categorical(df['healthcare_access'], categories=access_order, ordered=True)
df['economic_classification'] = pd.Categorical(df['economic_classification'], categories=economic_order, ordered=True)

#%% Cleaned Data
import os

os.makedirs('../data', exist_ok=True)

df.to_csv('../data/cleaned_colorectal.csv', index=False)

#%% Categorical Columns Summary
summary_table = PrettyTable()
summary_table.field_names = ["Column", "Unique Values"]

for col in cat_cols:
    summary_table.add_row([col, df[col].nunique()])

print("\nCategorical Feature Cardinality:")
print(summary_table.get_string(title="Cleaned Categorical Overview"))

#%% Bar Chart for Categorical Feature Cardinality

import matplotlib.pyplot as plt
import seaborn as sns

cat_cardinality = df.select_dtypes(include='object').nunique().sort_values(ascending=True)

plt.figure(figsize=(10, 8))
sns.barplot(x=cat_cardinality.values, y=cat_cardinality.index, palette='viridis')

plt.title('Categorical Feature Cardinality', fontsize=16, fontweight='bold', color='darkblue')
plt.xlabel('Number of Unique Values', fontsize=14, fontweight='bold', color='darkred')
plt.ylabel('Categorical Features', fontsize=14, fontweight='bold', color='darkred')
plt.grid(True, linestyle='--', linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()

#%% Visuals
import matplotlib.pyplot as plt

#%% Gender Distribution
df['gender'] = df['gender'].map({'f': 'Female', 'm': 'Male'})

gender_counts = df['gender'].value_counts()
labels = gender_counts.index
sizes = gender_counts.values
colors = ['#66c2a5', '#fc8d62']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, autopct='%.2f%%', startangle=90, colors=colors, textprops={'fontsize': 12})
plt.title('Gender Distribution', fontdict={'fontsize': 16, 'fontweight': 'bold', 'color': 'darkblue'})
plt.tight_layout()
plt.show()

#%% Mortality by Treatment Type (Count Plot)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='treatment_type', hue='mortality', palette='Set2')

plt.title('Mortality by Treatment Type', fontdict={'fontsize': 16, 'fontweight': 'bold', 'color': 'darkblue'})
plt.xlabel('Treatment Type', fontsize=14)
plt.ylabel('Patient Count', fontsize=14)
plt.legend(title='Mortality', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Screening History (Bar)
screening_counts = df['screening_history'].value_counts().sort_index()

plt.figure(figsize=(7, 5))
sns.barplot(x=screening_counts.index, y=screening_counts.values, palette='coolwarm')
plt.title('Screening History Distribution', fontdict={'fontsize': 16, 'fontweight': 'bold'})
plt.xlabel('Screening History', fontsize=14)
plt.ylabel('Number of Patients', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()


#%% Survival by Cancer Stage
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='cancer_stage', hue='survival_5_years', palette='muted')

plt.title('5-Year Survival by Cancer Stage', fontdict={'fontsize': 16, 'fontweight': 'bold'})
plt.xlabel('Cancer Stage', fontsize=14)
plt.ylabel('Patient Count', fontsize=14)
plt.legend(title='Survived 5 Years', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

#%% Subplot

# Treatment Type by Gender and Stage
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Treatment Type Patterns by Gender and Cancer Stage', fontsize=18, fontweight='bold')

#1. Treatment by Gender
sns.countplot(data=df, x='treatment_type', hue='gender', ax=axes[0, 0], palette='pastel')
axes[0, 0].set_title('Treatment Type by Gender')
axes[0, 0].grid(True, linestyle='--', linewidth=0.5)

#2. Treatment by Cancer Stage
sns.countplot(data=df, x='treatment_type', hue='cancer_stage', ax=axes[0, 1], palette='Set3')
axes[0, 1].set_title('Treatment Type by Cancer Stage')
axes[0, 1].grid(True, linestyle='--', linewidth=0.5)

#3. Survival by Treatment
sns.countplot(data=df, x='treatment_type', hue='survival_5_years', ax=axes[1, 0], palette='coolwarm')
axes[1, 0].set_title('Survival by Treatment Type')
axes[1, 0].grid(True, linestyle='--', linewidth=0.5)

#4. Mortality by Treatment
sns.countplot(data=df, x='treatment_type', hue='mortality', ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Mortality by Treatment Type')
axes[1, 1].grid(True, linestyle='--', linewidth=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


#%% Outlier Detection

# Used IQR and Z-Score Methods
# Focusing on continuous variables (Age & BMI)
df_raw = df.copy()

#%%
numeric_cols = [
    'age',
    'tumor_size_mm',
    'healthcare_costs',
    'incidence_rate_per_100k',
    'mortality_rate_per_100k'
]

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    print(f"{column}: Removed {len(outliers)} outliers")
    return cleaned_df

for col in numeric_cols:
    df = remove_outliers_iqr(df, col)

#%% Visuals for IQR Outlier Removal (Boxplot)
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col], color='lightblue')
    plt.title(f'Box Plot of {col} (Post-Outlier Removal)')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

#%% Subplot (IQR)

fig, axes = plt.subplots(nrows=1, ncols=len(numeric_cols), figsize=(20, 4))

for i, col in enumerate(numeric_cols):
    sns.boxplot(x=df[col], ax=axes[i], color='lightblue')
    axes[i].set_title(f'{col}\n(IQR Cleaned)')
    axes[i].set_xlabel(col)

plt.tight_layout()
plt.show()


#%% Z-Score Function
from scipy.stats import zscore

def remove_outliers_zscore(df, column, threshold=3):
    z_scores = zscore(df[column])
    mask = abs(z_scores) < threshold
    removed = len(df) - sum(mask)
    print(f"{column}: Removed {removed} outliers using Z-score")
    return df[mask]

#%% Adding Numeric Columns
numeric_cols = [
    'age',
    'tumor_size_mm',
    'healthcare_costs',
    'incidence_rate_per_100k',
    'mortality_rate_per_100k'
]

df_zscore = df_raw.copy()

for col in numeric_cols:
    df_zscore = remove_outliers_zscore(df_zscore, col)

#%%

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_zscore[col], color='lightgreen')
    plt.title(f'Box Plot of {col} (Post Z-score Removal)')
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

#%% Subplots (Z-Score)
fig, axes = plt.subplots(nrows=1, ncols=len(numeric_cols), figsize=(20, 4))

for i, col in enumerate(numeric_cols):
    sns.boxplot(x=df_zscore[col], ax=axes[i], color='lightgreen')
    axes[i].set_title(f'{col}\n(Cleaned Z-score)')
    axes[i].set_xlabel(col)

plt.tight_layout()
plt.show()


#%% Normality Testing
from scipy.stats import shapiro, anderson, kstest, norm
import statsmodels.api as sm

numeric_cols = [
    'age',
    'tumor_size_mm',
    'healthcare_costs',
    'incidence_rate_per_100k',
    'mortality_rate_per_100k'
]

#%% Normality Tests & Visuals

for col in numeric_cols:
    data = df[col].dropna()

    print(f"\n Normality Testing for: {col}")

    # Shapiro-Wilk Test
    stat_sw, p_sw = shapiro(data)
    print(f"Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.4f} → {'Normal' if p_sw > 0.05 else 'Not Normal'}")

    # Anderson-Darling Test
    result_ad = anderson(data)
    print(f"Anderson-Darling: A={result_ad.statistic:.4f}")
    for i in range(len(result_ad.critical_values)):
        sig_level = result_ad.significance_level[i]
        crit_val = result_ad.critical_values[i]
        print(f"  {sig_level}%: {crit_val:.4f} → {'Normal' if result_ad.statistic < crit_val else 'Not Normal'}")

    # Kolmogorov-Smirnov Test (against normal distribution)
    stat_ks, p_ks = kstest(data, 'norm', args=(data.mean(), data.std()))
    print(f"Kolmogorov-Smirnov: D={stat_ks:.4f}, p={p_ks:.4f} → {'Normal' if p_ks > 0.05 else 'Not Normal'}")

    # Hist
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(data, kde=True, color='skyblue')
    plt.title(f'Histogram of {col}')

    # QQ Plot
    plt.subplot(1, 2, 2)
    sm.qqplot(data, line='s', ax=plt.gca())
    plt.title(f'QQ Plot of {col}')
    plt.tight_layout()
    plt.show()









