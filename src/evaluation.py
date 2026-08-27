"""
Evaluation and visualization module for Crop Yield Prediction.
Generates interactive Plotly and Matplotlib charts for predictions, feature importance,
actual vs predicted comparisons, and historical trends.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from src.preprocessing import load_dataset, get_train_test_data


def get_feature_importances(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extracts and maps feature importance scores from the trained model pipeline.
    """
    regressor = pipeline.named_steps.get("regressor")
    preprocessor = pipeline.named_steps.get("preprocessor")

    if not hasattr(regressor, "feature_importances_"):
        return pd.DataFrame(columns=["Feature", "Importance", "Group"])

    # Extract transformed feature names
    feature_names = preprocessor.get_feature_names_out()
    importances = regressor.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    })

    # Group features by base domain
    def map_group(f_name):
        if "Area" in f_name:
            return "Country / Region"
        elif "Item" in f_name:
            return "Crop Variety"
        elif "average_rain_fall" in f_name or "rainfall" in f_name.lower():
            return "Rainfall"
        elif "pesticides" in f_name.lower():
            return "Pesticides"
        elif "temp" in f_name.lower():
            return "Temperature"
        elif "year" in f_name.lower():
            return "Year Trend"
        return "Environmental"

    fi_df["Group"] = fi_df["Feature"].apply(map_group)
    fi_df = fi_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return fi_df


def plot_feature_importance_top(pipeline: Pipeline, top_n: int = 15) -> go.Figure:
    """
    Generates a horizontal bar chart of the top N most important features.
    """
    fi_df = get_feature_importances(pipeline)
    if fi_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Feature importances not available for this model type", showarrow=False)
        return fig

    top_df = fi_df.head(top_n).iloc[::-1]  # Reverse for ascending horizontal display

    # Format feature names for clean UI display
    top_df["CleanFeature"] = (
        top_df["Feature"]
        .str.replace("cat__Area_", "Country: ")
        .str.replace("cat__Item_", "Crop: ")
        .str.replace("num__average_rain_fall_mm_per_year", "Rainfall (mm/yr)")
        .str.replace("num__pesticides_tonnes", "Pesticides (tonnes)")
        .str.replace("num__avg_temp", "Temperature (°C)")
        .str.replace("num__Year", "Year")
    )

    fig = go.Figure(
        go.Bar(
            x=top_df["Importance"] * 100,
            y=top_df["CleanFeature"],
            orientation="h",
            marker=dict(
                color=top_df["Importance"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Weight %"),
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Top {top_n} Predictive Feature Importances</b>",
        xaxis_title="Relative Importance Contribution (%)",
        yaxis_title="",
        template="plotly_dark",
        paper_bgcolor="rgba(13, 59, 37, 0.4)",
        plot_bgcolor="rgba(8, 43, 27, 0.6)",
        font=dict(color="#d8eadb"),
        height=450,
        margin=dict(l=20, r=20, t=50, b=40),
    )
    return fig


def plot_actual_vs_predicted(pipeline: Pipeline, sample_size: int = 1000) -> go.Figure:
    """
    Generates a scatter plot of Actual vs Predicted yield on held-out test data with a 45° reference line.
    """
    _, X_test, _, y_test, _ = get_train_test_data()

    if len(X_test) > sample_size:
        indices = np.random.RandomState(42).choice(len(X_test), size=sample_size, replace=False)
        X_sample = X_test.iloc[indices]
        y_sample = y_test.iloc[indices]
    else:
        X_sample = X_test
        y_sample = y_test

    y_pred = pipeline.predict(X_sample)

    min_val = min(y_sample.min(), y_pred.min())
    max_val = max(y_sample.max(), y_pred.max())

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=y_sample,
            y=y_pred,
            mode="markers",
            name="Predictions",
            marker=dict(
                size=6,
                color="#81c784",
                opacity=0.65,
                line=dict(width=0.5, color="#ffffff"),
            ),
            hovertemplate="Actual: %{x:,.0f} hg/ha<br>Predicted: %{y:,.0f} hg/ha<extra></extra>",
        )
    )

    # 45-degree perfect prediction line
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal 1:1 Line",
            line=dict(color="#ffca28", dash="dash", width=2),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="<b>Actual vs. Predicted Crop Yield (Validation Set)</b>",
        xaxis_title="Actual Yield (hg/ha)",
        yaxis_title="Predicted Yield (hg/ha)",
        template="plotly_dark",
        paper_bgcolor="rgba(13, 59, 37, 0.4)",
        plot_bgcolor="rgba(8, 43, 27, 0.6)",
        font=dict(color="#d8eadb"),
        height=450,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.3)"),
    )
    return fig


def plot_historical_and_forecast(
    pipeline: Pipeline,
    country: str,
    crop: str,
    selected_year: int,
    current_prediction_hg_ha: float,
    rainfall: float,
    pesticides: float,
    temperature: float,
) -> go.Figure:
    """
    Generates a timeline line chart of Historical yield (1990–2013) plus future projection up to selected_year.
    """
    df = load_dataset()
    hist_subset = df[(df["Area"] == country) & (df["Item"] == crop)]

    fig = go.Figure()

    if not hist_subset.empty:
        hist_trend = (
            hist_subset.groupby("Year", as_index=False)["hg/ha_yield"]
            .mean()
            .sort_values("Year")
        )
        # Historical actuals trace
        fig.add_trace(
            go.Scatter(
                x=hist_trend["Year"],
                y=hist_trend["hg/ha_yield"] / 10000.0,
                mode="lines+markers",
                name=f"Historical Actual ({country})",
                line=dict(color="#66bb6a", width=3),
                marker=dict(size=7, color="#2e7d32"),
                hovertemplate="Year: %{x}<br>Actual Yield: %{y:.2f} t/ha<extra></extra>",
            )
        )
        max_hist_year = int(hist_trend["Year"].max())
    else:
        max_hist_year = 2013

    # If selected year is beyond historical max
    if selected_year > max_hist_year:
        proj_years = list(range(max_hist_year, selected_year + 1))
        proj_inputs = pd.DataFrame({
            "Area": [country] * len(proj_years),
            "Item": [crop] * len(proj_years),
            "Year": proj_years,
            "average_rain_fall_mm_per_year": [rainfall] * len(proj_years),
            "pesticides_tonnes": [pesticides] * len(proj_years),
            "avg_temp": [temperature] * len(proj_years),
        })
        proj_preds = pipeline.predict(proj_inputs) / 10000.0

        fig.add_trace(
            go.Scatter(
                x=proj_years,
                y=proj_preds,
                mode="lines+markers",
                name="Model Forecast Projection",
                line=dict(color="#ffa726", width=3, dash="dot"),
                marker=dict(size=8, color="#ff9800"),
                hovertemplate="Year: %{x}<br>Forecast Yield: %{y:.2f} t/ha<extra></extra>",
            )
        )

    # Marker for current user prediction
    fig.add_trace(
        go.Scatter(
            x=[selected_year],
            y=[current_prediction_hg_ha / 10000.0],
            mode="markers",
            name="Current User Input Prediction",
            marker=dict(size=14, color="#e91e63", symbol="star"),
            hovertemplate="<b>Selected Input Target</b><br>Year: %{x}<br>Yield: %{y:.2f} t/ha<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Yield Trajectory for {crop} in {country} (1990 – {max(selected_year, max_hist_year)})</b>",
        xaxis_title="Year",
        yaxis_title="Crop Yield (tonnes/hectare)",
        template="plotly_dark",
        paper_bgcolor="rgba(13, 59, 37, 0.4)",
        plot_bgcolor="rgba(8, 43, 27, 0.6)",
        font=dict(color="#d8eadb"),
        height=420,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0.3)"),
    )
    return fig


def plot_metrics_summary(metrics: dict) -> go.Figure:
    """
    Generates a card-like bar visual summarizing R2, MAE, and RMSE.
    """
    if not metrics:
        return go.Figure()

    r2 = metrics.get("r2_test", 0.0)
    mae_t = metrics.get("mae", 0.0) / 10000.0
    rmse_t = metrics.get("rmse", 0.0) / 10000.0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="R² Score (%)",
            x=["R² Score (%)"],
            y=[r2 * 100],
            marker_color="#81c784",
            text=[f"{r2*100:.2f}%"],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="MAE (tonnes/ha)",
            x=["MAE (t/ha)"],
            y=[mae_t],
            marker_color="#4fc3f7",
            text=[f"{mae_t:.2f} t/ha"],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="RMSE (tonnes/ha)",
            x=["RMSE (t/ha)"],
            y=[rmse_t],
            marker_color="#ffb74d",
            text=[f"{rmse_t:.2f} t/ha"],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="<b>Key Model Evaluation Benchmarks</b>",
        template="plotly_dark",
        paper_bgcolor="rgba(13, 59, 37, 0.4)",
        plot_bgcolor="rgba(8, 43, 27, 0.6)",
        font=dict(color="#d8eadb"),
        height=320,
        margin=dict(l=20, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig
