import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def setup_styles():
    """Configures the global style for all plots."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_distribution(df: pd.DataFrame, column: str, title: str = None):
    """
    Plots the distribution of a single variable.
    - Histogram with KDE for numerical data.
    - Countplot for categorical data.
    """
    plt.figure(figsize=(8, 5))
    
    # Check if column is effectively categorical (few unique values)
    if df[column].nunique() < 10:
        ax = sns.countplot(x=column, data=df, palette='viridis')
        # Add counts on top of bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                        textcoords='offset points')
    else:
        sns.histplot(df[column], kde=True, color='teal')
    
    plt.title(title if title else f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame):
    """Plots a correlation heatmap for the entire dataframe."""
    plt.figure(figsize=(16, 12))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool)) # Mask the upper triangle
    
    sns.heatmap(corr, mask=mask, annot=False, fmt=".2f", cmap='coolwarm', 
                linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_target_correlations(df: pd.DataFrame, target: str, k: int = 10):
    """Plots a bar chart of the top k features correlated with the target."""
    corr = df.corr()[target].drop(target).sort_values(ascending=False)
    
    # Get top positive and bottom negative correlations
    top_pos = corr.head(k)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_pos.values, y=top_pos.index, palette='rocket')
    plt.title(f'Top {k} Features Correlated with {target}')
    plt.xlabel('Correlation Coefficient')
    plt.axvline(x=0, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_categorical_vs_target(df: pd.DataFrame, col: str, target: str):
    """
    Plots a normalized stacked bar chart to show the relationship 
    between a categorical feature and the binary target.
    """
    crosstab = pd.crosstab(df[col], df[target], normalize='index')
    
    crosstab.plot(kind='bar', stacked=True, colormap='coolwarm', figsize=(8, 5))
    plt.title(f'Relationship: {col} vs {target}')
    plt.ylabel('Proportion')
    plt.xlabel(col)
    plt.legend(title=target, loc='upper right')
    plt.tight_layout()
    plt.show()