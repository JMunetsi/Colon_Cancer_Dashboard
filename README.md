# Colorectal Cancer Data Visualization & Interactive Dashboard

Exploratory analysis and an interactive web dashboard built on a real-world colorectal cancer dataset - 167,000+ patient records and 28 clinical, demographic, lifestyle, and epidemiological features. Built for DATS6401 (Visualization of Complex Data), George Washington University.

## Overview

The project has two phases:

1. **Static exploratory analysis** - data cleaning, outlier detection, PCA, normality testing, correlation analysis, and a full set of visualizations (histograms, KDE plots, boxplots, violin plots, heatmaps, QQ-plots, 3D plots, cluster maps) with observations on cancer stage, tumor size, healthcare costs, and demographic/lifestyle risk factors.
2. **Interactive dashboard** - a multi-tab Dash app that lets users dynamically clean data, detect outliers, run PCA, perform normality tests, apply transformations, and generate custom numerical/categorical plots in real time, without re-running code. Deployed on Google Cloud Platform.

## Key Findings

- Tumor size increases with cancer stage, and mortality rises sharply in metastatic cases
- Age, tumor size, and healthcare costs are the main drivers of variance (via PCA)
- Clustering surfaces natural groupings aligned with cancer stage and patient characteristics
- Healthcare costs stay high across economic classes, pointing to a broad global burden
- Lifestyle factors (smoking, alcohol use, BMI) show varied distributions but contribute meaningfully to overall structure

## Dashboard

Multi-tab Dash app covering:
- Numerical & categorical plots
- Dimensionality reduction (PCA)
- Normality tests
- Outlier detection
- Data cleaning
- Data transformation
- Summary statistics

## Tech Stack

Python · Pandas · Seaborn · Matplotlib · SciPy · Dash · Plotly · scikit-learn (PCA) · Google Cloud Platform (deployment)

## Repository Structure

- `data_cleaning.py` - preprocessing: type coercion, missing value handling, category standardization *(formerly Stage_1.py)*
- `dashboard.py` - interactive Dash app: PCA, outlier detection, normality tests, transformations, plotting *(formerly Stage_2.py)*
- `colorectal_cancer_dataset.csv` - source dataset

## Full Report

The complete write-up - methodology, all 40+ figures, and detailed observations per analysis — is available on request.
