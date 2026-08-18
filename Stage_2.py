
# Imports & styling

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


pio.templates.default = "plotly_white"

def style_fig(fig, xlab=None, ylab=None):
    fig.update_layout(
        title_font=dict(family="serif", color="blue", size=16),
        xaxis_title=xlab,
        yaxis_title=ylab,
        xaxis_title_font=dict(family="serif", color="darkred", size=14),
        yaxis_title_font=dict(family="serif", color="darkred", size=14),
        xaxis=dict(showgrid=True, gridcolor="lightgrey"),
        yaxis=dict(showgrid=True, gridcolor="lightgrey"),
        legend_title_text=None
    )
    return fig


# ============================
# Stage 1 alignment: Load & clean data
# ============================
url = "https://github.com/JMunetsi/Colon_Cancer_Dashboard/raw/refs/heads/main/colorectal_cancer_dataset.csv"
df = pd.read_csv(url)


df.columns = df.columns.str.strip().str.lower()

for col in df.select_dtypes(include="object"):
    df[col] = df[col].astype(str).str.strip().str.lower()

num_cols_clean = [
    "age", "tumor_size_mm", "healthcare_costs",
    "incidence_rate_per_100k", "mortality_rate_per_100k"
]
for c in num_cols_clean:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "cancer_stage" in df.columns:
    df["cancer_stage"] = pd.Categorical(
        df["cancer_stage"], ["localized", "regional", "metastatic"], ordered=True
    )

if "obesity_bmi" in df.columns:
    df["obesity_bmi"] = pd.Categorical(
        df["obesity_bmi"], ["normal", "overweight", "obese"], ordered=True
    )

df_sample = df.sample(min(len(df), 1000), random_state=42)



# Feature lists (aligned with Stage 1)


num_features = [
    c for c in [
        "age",
        "tumor_size_mm",
        "healthcare_costs",
        "incidence_rate_per_100k",
        "mortality_rate_per_100k",
        "log_costs",
        "log_age"
    ] if c in df.columns
]

cat_features = [
    c for c in [
        "gender",
        "cancer_stage",
        "family_history",
        "smoking_history",
        "alcohol_consumption",
        "diabetes",
        "inflammatory_bowel_disease",
        "genetic_mutation",
        "treatment_type",
        "insurance_status",
        "urban_or_rural",
        "country",
        "screening_history",
        "early_detection",
        "survival_5_years",
        "mortality",
        "economic_classification",
        "healthcare_access",
        "diet_risk",
        "physical_activity",
        "obesity_bmi"
    ] if c in df.columns
]



# App initialization & base layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
app.title = "Colorectal Cancer Dashboard"

app.layout = dbc.Container([
    html.H2("Colorectal Cancer Interactive Dashboard", className="text-center mt-4"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader("Dataset Info"),
            dbc.CardBody([
                html.H5("Rows / Columns", className="card-title"),
                html.P(f"{len(df)} rows × {len(df.columns)} columns", className="card-text"),
                dbc.Button("Reload sample", id="btn-reload-sample", color="warning", outline=True)
            ])
        ]), width=4),


        dbc.Col(dbc.Card([
            dbc.CardHeader("Data Preview"),
            dbc.CardBody([
                html.Div(
                    id="preview-div",
                    children=[dbc.Table.from_dataframe(df.head(3), striped=True, bordered=True, hover=True)]
                )
            ])
        ]), width=4),


    ], className="mb-4"),

    dcc.Tabs(
        id="tabs",
        value="tab-numerical",
        children=[
            dcc.Tab(label="Numerical Plots", value="tab-numerical"),
            dcc.Tab(label="Categorical Plots", value="tab-categorical"),
            dcc.Tab(label="Dimensionality Reduction", value="tab-dimred"),
            dcc.Tab(label="Normality Tests", value="tab-normality"),
            dcc.Tab(label="Outlier Detection", value="tab-outliers"),
            dcc.Tab(label="Data Cleaning", value="tab-cleaning"),
            dcc.Tab(label="Data Transformation", value="tab-transform"),
            dcc.Tab(label="Statistics", value="tab-stats"),
        ],
        style={
            "backgroundColor": "#2c2c2c",
            "color": "white"
        },
        colors={
            "border": "white",
            "primary": "gold",
            "background": "#2c2c2c"
        }
    ),

    html.Br(),
    html.Div(id="tab-content")
], fluid=True)

@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):

    # TAB: NUMERICAL PLOTS

    if tab == "tab-numerical":

        age_min = int(df["age"].min())
        age_max = int(df["age"].max())
        age_mid = (age_min + age_max) // 2

        return dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Numerical Plot Controls"),
                    dbc.CardBody([

                        dbc.Label("Select Feature"),
                        dcc.Dropdown(
                            id="num-feature",
                            options=[{"label": f, "value": f} for f in num_features],
                            value=num_features[0] if num_features else None,
                            clearable=False,
                            style={"marginBottom": "15px", "backgroundColor": "white",
                                "color": "black",
                                "borderColor": "#444",
                                }
                        ),

                        dbc.Label("Plot Type"),
                        dcc.RadioItems(
                            id="num-plot-type",
                            options=[
                                {"label": "Histogram", "value": "hist"},
                                {"label": "Box", "value": "box"},
                                {"label": "Violin", "value": "violin"},
                                {"label": "Scatter vs Tumor Size", "value": "scatter"}
                            ],
                            value="hist",
                            inline=True,
                            style={"marginBottom": "20px"}
                        ),

                        html.Hr(),

                        dbc.Label("Bins (for histogram)", className="mt-3 mb-2"),
                        dcc.Slider(
                            id="num-bins",
                            min=5, max=200, step=5, value=30,
                            marks={i: str(i) for i in range(0, 201, 20)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),

                        dbc.Label("Filter Range (Age scale)", className="mt-4 mb-2"),
                        dcc.RangeSlider(
                            id="num-range",
                            min=age_min,
                            max=age_max,
                            step=1,
                            value=[age_min, age_max],
                            marks={
                                age_min: str(age_min),
                                age_mid: "Mid",
                                age_max: str(age_max)
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ])
                ]),
                width=4
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Numerical Plot"),
                    dbc.CardBody([
                        dcc.Loading(dcc.Graph(id="num-graph"))
                    ])
                ]),
                width=8
            )
        ])


    # TAB: CATEGORICAL PLOTS

    elif tab == "tab-categorical":
        return dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Categorical Plot Controls"),
                    dbc.CardBody([
                        dbc.Label("Select Feature"),
                        dcc.Dropdown(
                            id="cat-feature",
                            options=[{"label": c, "value": c} for c in cat_features],
                            value=cat_features[0] if cat_features else None,
                            clearable=False,
                            style={"marginBottom": "15px", "backgroundColor": "white",
                                   "color": "black",
                                   "borderColor": "#444",}

                        ),

                        dbc.Label("Plot Type", className="mt-2"),
                        dcc.RadioItems(
                            id="cat-plot-type",
                            options=[
                                {"label": "Bar", "value": "bar"},
                                {"label": "Pie", "value": "pie"}
                            ],
                            value="bar",
                            inline=True
                        )
                    ])
                ]),
                width=4
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Categorical Plot"),
                    dbc.CardBody([
                        dcc.Graph(id="cat-graph"),
                        html.Div(id="cat-table")
                    ])
                ]),
                width=8
            )
        ])


    # TAB: DIMENSIONALITY REDUCTION

    elif tab == "tab-dimred":
        return dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Dimensionality Reduction Controls"),
                    dbc.CardBody([
                        dbc.Label("Method"),
                        dcc.RadioItems(
                            id="dr-method",
                            options=[
                                {"label": "PCA", "value": "pca"},
                                {"label": "t-SNE", "value": "tsne"}
                            ],
                            value="pca",
                            inline=True
                        ),

                        dbc.Label("Color By"),
                        dcc.Dropdown(
                            id="dr-color",
                            options=[{"label": c, "value": c} for c in cat_features],
                            value=cat_features[0] if cat_features else None,
                            clearable=False,
                            style={"marginBottom": "15px", "backgroundColor": "white",
                                   "color": "black",
                                   "borderColor": "#444", }
                        )
                    ])
                ]),
                width=4
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Dimensionality Reduction Plots"),
                    dbc.CardBody([
                        dcc.Loading(dcc.Graph(id="dr-graph")),
                        dcc.Graph(id="dr-scree")
                    ])
                ]),
                width=8
            )
        ])

    # TAB: NORMALITY TESTS

    elif tab == "tab-normality":
        return dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Normality Test Controls"),
                    dbc.CardBody([
                        dbc.Label("Select Feature"),
                        dcc.Dropdown(
                            id="norm-feature",
                            options=[{"label": f, "value": f} for f in num_features],
                            value=num_features[0] if num_features else None,
                            clearable=False,
                            style={"marginBottom": "15px", "backgroundColor": "white",
                                   "color": "black",
                                   "borderColor": "#444",}
                        ),

                        dbc.Label("Select Methods"),
                        dcc.Checklist(
                            id="norm-methods",
                            options=[
                                {"label": "Shapiro-Wilk", "value": "shapiro"},
                                {"label": "Anderson-Darling", "value": "anderson"},
                                {"label": "Kolmogorov-Smirnov", "value": "ks"}
                            ],
                            value=["shapiro"]
                        ),

                        dbc.Label("Plot Type", className="mt-2"),
                        dcc.RadioItems(
                            id="norm-plot-type",
                            options=[
                                {"label": "Histogram", "value": "hist"},
                                {"label": "QQ Plot", "value": "qq"}
                            ],
                            value="hist",
                            inline=True
                        )
                    ])
                ]),
                width=4
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Normality Test Results"),
                    dbc.CardBody([
                        dcc.Graph(id="norm-graph"),
                        html.Div(id="norm-results")
                    ])
                ]),
                width=8
            )
        ])


    # TAB: OUTLIER DETECTION

    elif tab == "tab-outliers":
        return dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Outlier Detection Controls"),
                    dbc.CardBody([
                        dbc.Label("Select Feature"),
                        dcc.Dropdown(
                            id="outlier-feature",
                            options=[{"label": f, "value": f} for f in num_features],
                            value=num_features[0] if num_features else None,
                            clearable=False,
                            style={
                                "backgroundColor": "white",
                                "color": "black",
                                "borderColor": "#444",
                                "marginBottom": "15px"
                            }

                        ),

                        dbc.Label("Method", className="mt-2"),
                        dcc.RadioItems(
                            id="outlier-method",
                            options=[
                                {"label": "IQR", "value": "iqr"},
                                {"label": "Z-score", "value": "zscore"}
                            ],
                            value="iqr",
                            inline=True
                        ),

                        dbc.Label("Threshold", className="mt-2"),
                        dcc.Slider(
                            id="outlier-thresh",
                            min=1.0,
                            max=4.0,
                            step=0.1,
                            value=1.5,
                            marks={
                                1.0: "1.0",
                                2.0: "2.0",
                                3.0: "3.0",
                                4.0: "4.0"
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ])
                ]),
                width=4
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Outlier Plot"),
                    dbc.CardBody([dcc.Graph(id="outlier-graph")])
                ]),
                width=8
            )
        ])

    # TAB: DATA CLEANING

    elif tab == "tab-cleaning":
        return dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Data Cleaning Controls"),
                    dbc.CardBody([
                        dcc.Checklist(
                            id="clean-methods",
                            options=[
                                {"label": "Strip/Lowercase", "value": "strip_lower"},
                                {"label": "Drop Duplicates", "value": "drop_dupes"},
                                {"label": "Fill Missing (numeric mean)", "value": "fill_missing"}
                            ],
                            value=[]
                        ),
                        html.Br(),
                        dbc.Button("Apply Cleaning", id="clean-apply", n_clicks=0, color="primary")
                    ])
                ]),
                width=4
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Cleaned Data Preview"),
                    dbc.CardBody([html.Div(id="clean-preview")])
                ]),
                width=8
            )
        ])


    # TAB: DATA TRANSFORMATION

    elif tab == "tab-transform":
        return dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Transformation Controls"),
                    dbc.CardBody([
                        dbc.Label("Select Transformation"),
                        dcc.Checklist(
                            id="transform-ops",
                            options=[
                                {"label": "Log healthcare_costs", "value": "log"},
                                {"label": "Standardize healthcare_costs", "value": "standardize"},
                                {"label": "Normalize healthcare_costs", "value": "normalize"}
                            ],
                            value=[]
                        ),
                        html.Br(),
                        dbc.Label("Note: Transformations apply only to numeric features.")
                    ])
                ]),
                width=4
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Transformed Plot"),
                    dbc.CardBody([
                        dcc.Graph(id="transform-graph")
                    ])
                ]),
                width=8
            )
        ])


    # TAB: SUMMARY STATISTICS

    elif tab == "tab-stats":
        return dbc.Row([

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Statistics Controls"),
                    dbc.CardBody([
                        dbc.Label("Select Feature"),
                        dcc.Dropdown(
                            id="stats-feature",
                            options=[{"label": f, "value": f} for f in num_features],
                            value=num_features[0] if num_features else None,
                            clearable=False,
                            style={"marginBottom": "15px", "backgroundColor": "white",
                                   "color": "black",
                                   "borderColor": "#444",}
                        ),

                        dbc.Label("Select Metrics", className="mt-3"),
                        dcc.Checklist(
                            id="stats-metrics",
                            options=[
                                {"label": "Count", "value": "count"},
                                {"label": "Mean", "value": "mean"},
                                {"label": "Median", "value": "median"},
                                {"label": "Std Dev", "value": "std"},
                                {"label": "Min", "value": "min"},
                                {"label": "Max", "value": "max"},
                                {"label": "25% (Q1)", "value": "25%"},
                                {"label": "75% (Q3)", "value": "75%"}
                            ],
                            value=["count", "mean", "median", "std", "min", "max"]
                        )
                    ])
                ]),
                width=4
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Summary Statistics"),
                    dbc.CardBody([
                        html.Div(id="stats-table")
                    ])
                ]),
                width=8
            ),

            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Storytelling Subplot"),
                    dbc.CardBody([
                        dcc.Graph(id="story-graph")
                    ])
                ]),
                width=12
            )
        ])

    # FALLBACK

    else:
        return html.Div("Tab not found")


# Numerical Plots Callback

@app.callback(
    Output("num-graph", "figure"),
    [
        Input("num-feature", "value"),
        Input("num-plot-type", "value"),
        Input("num-bins", "value"),
        Input("num-range", "value")
    ]
)
def update_numerical(feature, plot_type, bins, range_vals):

    if feature is None or feature not in df.columns:
        return style_fig(px.scatter(title="No numeric feature selected"))

    if not pd.api.types.is_numeric_dtype(df[feature]):
        return style_fig(px.scatter(title=f"{feature} is not numeric"))

    dff = df.copy()

    if range_vals:
        lo, hi = range_vals
        dff = dff[(dff["age"] >= lo) & (dff["age"] <= hi)]

    if plot_type == "hist":
        fig = px.histogram(
            dff, x=feature, nbins=bins,
            color="cancer_stage" if "cancer_stage" in dff.columns else None,
            title=f"Histogram of {feature}"
        )
        return style_fig(fig, xlab=feature, ylab="Count")

    elif plot_type == "box":
        fig = px.box(
            dff, y=feature,
            color="cancer_stage" if "cancer_stage" in dff.columns else None,
            title=f"Box Plot of {feature}"
        )
        return style_fig(fig, ylab=feature)

    elif plot_type == "violin":
        fig = px.violin(
            dff, y=feature, box=True, points="all",
            color="cancer_stage" if "cancer_stage" in dff.columns else None,
            title=f"Violin Plot of {feature}"
        )
        return style_fig(fig, ylab=feature)

    else:
        if "tumor_size_mm" in dff.columns:
            fig = px.scatter(
                dff, x=feature, y="tumor_size_mm",
                color="cancer_stage" if "cancer_stage" in dff.columns else None,
                title=f"{feature} vs Tumor Size"
            )
            return style_fig(fig, xlab=feature, ylab="Tumor Size (mm)")
        else:
            fig = px.scatter(dff, x=feature, y=dff.index, title=f"Scatter Plot of {feature}")
            return style_fig(fig, xlab=feature, ylab="Index")


# Categorical Plots Callback

@app.callback(
    [Output("cat-graph", "figure"),
     Output("cat-table", "children")],
    [
        Input("cat-feature", "value"),
        Input("cat-plot-type", "value")
    ]
)
def update_categorical(feature, plot_type):

    if feature is None or feature not in df.columns:
        return style_fig(px.scatter(title="No categorical feature selected")), html.Div("No data")

    series = df[feature].astype("object")
    series = series.fillna("Missing")

    counts = (
        series
        .value_counts()
        .reset_index()
    )
    counts.columns = ["Category", "Count"]
    counts = counts.sort_values("Count", ascending=False)

    if counts.empty:
        return style_fig(px.scatter(title=f"No data in {feature}")), html.Div("Column is empty")

    if plot_type == "bar":
        fig = px.bar(
            counts,
            x="Category",
            y="Count",
            text="Count",
            title=f"Bar Plot – {feature}"
        )

    else:
        fig = px.pie(
            counts,
            names="Category",
            values="Count",
            title=f"Pie Chart – {feature}",
            hole=0.3
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )

    table = dbc.Table.from_dataframe(
        counts,
        striped=True,
        bordered=True,
        hover=True
    )

    return style_fig(fig), table


# Dimensionality Reduction Callback

@app.callback(
    [Output("dr-graph", "figure"),
     Output("dr-scree", "figure")],
    [
        Input("dr-method", "value"),
        Input("dr-color", "value")
    ]
)
def update_dimred(method, color):

    features = num_features
    if not features:
        return style_fig(px.scatter(title="No numeric features")), style_fig(px.bar(title="No scree"))

    df_clean = df[features].dropna()

    if df_clean.shape[0] < 5:
        return style_fig(px.scatter(title="Not enough numeric data")), style_fig(px.bar(title="No scree"))

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean)

    #PCA
    if method == "pca":
        pca = PCA(n_components=2)
        comps = pca.fit_transform(scaled)

        df_vis = df_clean.copy()
        df_vis["PC1"] = comps[:, 0]
        df_vis["PC2"] = comps[:, 1]

        if color in df.columns:
            df_vis[color] = df.loc[df_clean.index, color]

        fig_scatter = px.scatter(
            df_vis, x="PC1", y="PC2",
            color=color if color in df_vis.columns else None,
            title="PCA Scatter Plot"
        )
        fig_scatter = style_fig(fig_scatter, xlab="PC1", ylab="PC2")

        scree_vals = pca.explained_variance_ratio_
        fig_scree = px.bar(
            x=["PC1", "PC2"], y=scree_vals,
            title="PCA Scree Plot"
        )
        fig_scree = style_fig(fig_scree, xlab="Component", ylab="Explained Variance Ratio")

        return fig_scatter, fig_scree

    # t-SNE
    else:
        sample_n = min(800, df_clean.shape[0])
        df_small = df_clean.sample(sample_n, random_state=42)
        scaled_small = scaler.fit_transform(df_small)

        perp = min(30, max(5, sample_n // 10))

        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
            init="random"
        )
        comps = tsne.fit_transform(scaled_small)

        df_vis = df_small.copy()
        df_vis["TSNE1"] = comps[:, 0]
        df_vis["TSNE2"] = comps[:, 1]

        if color in df.columns:
            df_vis[color] = df.loc[df_small.index, color]

        fig_scatter = px.scatter(
            df_vis, x="TSNE1", y="TSNE2",
            color=color if color in df_vis.columns else None,
            title="t-SNE Scatter Plot"
        )
        fig_scatter = style_fig(fig_scatter, xlab="t-SNE 1", ylab="t-SNE 2")

        fig_scree = px.bar(
            x=["t-SNE"], y=[1],
            title="t-SNE has no explained variance ratio"
        )
        fig_scree = style_fig(fig_scree, xlab="Method", ylab="Dummy Value")

        return fig_scatter, fig_scree


# Normality Tests Callback
@app.callback(
    [Output("norm-graph", "figure"),
     Output("norm-results", "children")],
    [
        Input("norm-feature", "value"),
        Input("norm-methods", "value"),
        Input("norm-plot-type", "value")
    ]
)
def update_normality(feature, methods, plot_type):

    if feature is None or feature not in df.columns:
        return style_fig(px.scatter(title="No feature selected")), html.Div("No results")

    series = df[feature].dropna()

    if series.empty or series.nunique() < 2:
        return style_fig(px.scatter(title="Not enough data")), html.Div("Not enough data")

    methods = methods or []

    # Plot
    if plot_type == "hist":
        fig = px.histogram(
            x=series,
            nbins=30,
            title=f"Histogram of {feature}"
        )
        fig = style_fig(fig, xlab=feature, ylab="Count")

    else:
        osm, osr = stats.probplot(series, dist="norm")[0]
        fig = px.scatter(x=osm, y=osr, title=f"QQ Plot of {feature}")

        slope, intercept = np.polyfit(osm, osr, 1)
        fig.add_shape(
            type="line",
            x0=min(osm), y0=slope * min(osm) + intercept,
            x1=max(osm), y1=slope * max(osm) + intercept,
            line=dict(color="red", dash="dash")
        )
        fig = style_fig(fig, xlab="Theoretical Quantiles", ylab="Sample Quantiles")

    # Statistical results
    results = []

    if "shapiro" in methods:
        s = series.sample(5000) if len(series) > 5000 else series
        stat, p = stats.shapiro(s)
        results.append(html.Div(f"Shapiro-Wilk: W={stat:.4f}, p={p:.4e}"))

    if "anderson" in methods:
        res = stats.anderson(series, dist="norm")
        results.append(html.Div(f"Anderson-Darling: A2={res.statistic:.4f}"))

    if "ks" in methods:
        s_std = (series - series.mean()) / series.std(ddof=0)
        s_std = s_std.dropna()
        if len(s_std) > 0:
            stat, p = stats.kstest(s_std, "norm")
            results.append(html.Div(f"Kolmogorov-Smirnov: D={stat:.4f}, p={p:.4e}"))

    if not results:
        results = [html.Div("No methods selected.")]

    return fig, results

# Outlier Detection Callback

@app.callback(
    Output("outlier-graph", "figure"),
    [
        Input("outlier-feature", "value"),
        Input("outlier-method", "value"),
        Input("outlier-thresh", "value")
    ]
)
def detect_outliers(feature, method, thresh):

    if feature is None or feature not in df.columns:
        return style_fig(px.scatter(title="No feature selected"))

    if not pd.api.types.is_numeric_dtype(df[feature]):
        return style_fig(px.scatter(title=f"{feature} is not numeric"))

    series = df[feature].dropna()

    if series.empty:
        return style_fig(px.scatter(title="No data available"))

    if method == "zscore":
        if series.std() == 0:
            return style_fig(px.scatter(title="Standard deviation is zero — no outliers"))
        z = (series - series.mean()) / series.std()
        out_idx = z[z.abs() > thresh].index

    else:
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo = q1 - thresh * iqr
        hi = q3 + thresh * iqr
        out_idx = series[(series < lo) | (series > hi)].index

    fig = px.box(
        df, y=feature, points="all",
        color="cancer_stage" if "cancer_stage" in df.columns else None,
        title=f"Outlier Detection – {feature}"
    )

    if len(out_idx) > 0:
        fig.add_scatter(
            x=[feature] * len(out_idx),
            y=df.loc[out_idx, feature],
            mode="markers",
            name="Outliers",
            marker=dict(size=9, symbol="x", color="red")
        )

    return style_fig(fig, ylab=feature)

# Data Cleaning Callback
@app.callback(
    Output("clean-preview", "children"),
    [Input("clean-apply", "n_clicks"), Input("btn-reload-sample", "n_clicks")],
    [State("clean-methods", "value")],
    prevent_initial_call=False
)
def apply_cleaning(n_apply, n_reload, methods):

    methods = methods or []
    df_clean = df.copy()


    if "strip_lower" in methods:
        for col in df_clean.select_dtypes(include="object"):
            df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

    if "drop_dupes" in methods:
        df_clean = df_clean.drop_duplicates()

    if "fill_missing" in methods:
        df_clean = df_clean.fillna(df_clean.mean(numeric_only=True))

    return dbc.Table.from_dataframe(
        df_clean.head(10),
        striped=True, bordered=True, hover=True
    )


# Data Transformation Callback

@app.callback(
    Output("transform-graph", "figure"),
    Input("transform-ops", "value")
)
def transform_data(ops):

    if "healthcare_costs" not in df.columns:
        return style_fig(px.scatter(title="Column healthcare_costs not found"))

    ops = ops or []
    series = df["healthcare_costs"].dropna().astype(float)
    transformed = series.copy().values

    if "log" in ops:
        transformed = np.log1p(np.clip(transformed, a_min=0, a_max=None))

    if "standardize" in ops:
        transformed = StandardScaler().fit_transform(
            transformed.reshape(-1, 1)
        ).flatten()

    if "normalize" in ops:
        transformed = MinMaxScaler().fit_transform(
            transformed.reshape(-1, 1)
        ).flatten()

    fig = px.histogram(
        x=transformed,
        nbins=30,
        title="Transformed Healthcare Costs"
    )

    return style_fig(fig, xlab="Value", ylab="Count")


# Statistics + Storytelling Callback

@app.callback(
    [Output("stats-table", "children"),
     Output("story-graph", "figure")],
    [
        Input("stats-feature", "value"),
        Input("stats-metrics", "value")
    ]
)
def show_stats(feature, metrics):

    if feature is None or feature not in df.columns:
        return html.Div("No feature selected"), style_fig(px.scatter(title="No data"))

    if not pd.api.types.is_numeric_dtype(df[feature]):
        return html.Div("Feature must be numeric"), style_fig(px.scatter(title="Invalid feature"))

    vals = df[feature].dropna()
    desc = vals.describe().to_dict()

    metrics = metrics or []
    rows = []

    for m in metrics:
        if m == "median":
            v = vals.median()
        elif m == "25%":
            v = desc.get("25%")
        elif m == "75%":
            v = desc.get("75%")
        else:
            v = desc.get(m)

        rows.append({"Metric": m, "Value": round(float(v), 2) if v is not None else None})

    stats_df = pd.DataFrame(rows)
    stats_table = dbc.Table.from_dataframe(stats_df, striped=True, bordered=True, hover=True)

    # Storytelling subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Age Distribution",
            "Tumor Size by Stage",
            "Healthcare Costs by Economic Class",
            "Mortality vs Incidence"
        ]
    )

    # Age distribution
    if "age" in df.columns:
        fig.add_trace(go.Histogram(x=df["age"].dropna(), name="Age"), row=1, col=1)

    # Tumor size by stage
    if "tumor_size_mm" in df.columns and "cancer_stage" in df.columns:
        fig.add_trace(
            go.Box(x=df["cancer_stage"], y=df["tumor_size_mm"], name="Tumor Size"),
            row=1, col=2
        )

    # Healthcare costs by economic class
    if "economic_classification" in df.columns and "healthcare_costs" in df.columns:
        costs = df.groupby("economic_classification")["healthcare_costs"].mean().reset_index()
        fig.add_trace(
            go.Bar(x=costs["economic_classification"], y=costs["healthcare_costs"], name="Costs"),
            row=2, col=1
        )

    # Mortality vs incidence
    if "incidence_rate_per_100k" in df.columns and "mortality_rate_per_100k" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["incidence_rate_per_100k"],
                y=df["mortality_rate_per_100k"],
                mode="markers",
                name="Mortality vs Incidence"
            ),
            row=2, col=2
        )

    fig.update_layout(title_text="Storytelling Subplot", height=800)

    return stats_table, style_fig(fig)


if __name__ == "__main__":
    app.run(debug=False, port=8080, host="127.0.0.1")

