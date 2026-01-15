import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from IPython.display import display

def check_duplicates(df: pd.DataFrame):
    """
    Checks for duplicate rows. 
    If found, prints the count and displays the first 5 examples.
    """
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    if duplicates > 0:
        print("Duplicate rows found. Displaying first 5:")
        display(df[df.duplicated()].head())
    else:
        print(" No duplicate rows found.")



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


def plot_outlier_check(df: pd.DataFrame, column: str):
    """
    Displays a Box Plot to visually detect outliers in numerical data.
    """
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df[column], color='lightgreen')
    plt.title(f'Box Plot of {column} (Outlier Detection)')
    plt.tight_layout()
    plt.show()

def plot_numerical_vs_target(df: pd.DataFrame, col: str, target: str):
    """
    Plots a boxplot comparing a numerical feature against the target classes.
    """
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=target, y=col, data=df, palette='Set2', hue=target, legend=False)
    plt.title(f'{col} vs {target}')
    plt.tight_layout()
    plt.show()

def plot_categorical_grid(df: pd.DataFrame, target: str, exclude_cols: list = None):
    """
    Plots a grid of countplots for all categorical features against the target.
    """
    if exclude_cols is None:
        exclude_cols = []
        
    # Select categorical columns, excluding target and specified exclusions
    cat_cols = [col for col in df.columns if col not in exclude_cols and col != target]
    
    # Calculate grid dimensions dynamically (or fixed 4x4 as per original)
    n_cols = 4
    n_rows = (len(cat_cols) - 1) // n_cols + 1
    
    plt.figure(figsize=(20, 5 * n_rows))
    
    for i, col in enumerate(cat_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.countplot(x=col, hue=target, data=df, palette='coolwarm')
        plt.title(f'{col} vs {target}')
        plt.legend(title='Diabetes Risk', loc='upper right', fontsize='small')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Plots a correlation heatmap with annotations and a Red-Blue color scheme.
    """
    plt.figure(figsize=(14, 10))
    
    # Calculate correlation
    # We select only numerical columns to avoid errors if strings are present
    corr = df.select_dtypes(include=[np.number]).corr()
    
    
    sns.heatmap(
        corr, 
        mask=None,          
        annot=True,         # Show numbers 
        fmt=".2f",          # 2 decimal places
        cmap='RdBu_r',      # Red-Blue Reversed Color map 
        vmin=-1, vmax=1,    # Fixed scale from -1 to 1 
        linewidths=0.5, 
        cbar_kws={"shrink": .8}
    )
    
    plt.title('Correlation Heatmap (Feature vs Feature)', fontsize=16)
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

