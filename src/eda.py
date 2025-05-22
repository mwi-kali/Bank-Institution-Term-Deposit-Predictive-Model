import numpy as np
import pandas as pd
import plotly.express as px

from .config import MONTH_ORDER, DAY_ORDER, EDU_ORDER, MARITAL_ORDER, POUTCOME_ORDER, TERNARY_ORDER, BINARY_ORDER


def correlation_matrix(df: pd.DataFrame, cols):
    corr = df[cols].corr()
    return px.imshow(
        corr, text_auto='.2f', aspect='auto',
        color_continuous_scale='RdBu',
        title="Correlation Matrix"
    )

def numeric_distribution(df: pd.DataFrame, numeric_cols):
    figs = {}
    for c in numeric_cols:
        figs[c] = px.histogram(
            df, x=c, nbins=50, marginal='box',
            title=f"Distribution of {c}",
            labels={c:c, 'count':'Count'}
        )
    return figs

def ordered_category_freq(df: pd.DataFrame, col, order):
    return px.histogram(
        df, x=col,
        category_orders={col: order},
        title=f"Frequency of {col}",
        labels={col:col,'count':'Count'},
        nbins=len(order)
    )

def subscription_rate_by_category(df: pd.DataFrame, col, order):
    rate = (
        df.groupby(col)['y']
          .value_counts(normalize=True)
          .rename('rate')
          .reset_index()
          .query("y=='yes'")
    )
    return px.bar(
        rate, x=col, y='rate',
        category_orders={col:order},
        title=f"Subscription Rate by {col}",
        labels={'rate':'Subscription Rate'},
        text=rate['rate'].map("{:.1%}".format)
    )

def unknown_placeholder_counts(df: pd.DataFrame):
    counts = (df=='unknown').sum()
    counts = counts[counts>0]
    return px.bar(
        x=counts.index, y=counts.values,
        title="'unknown' Placeholder Count by Feature",
        labels={'x':'Feature','y':'Count'}
    )