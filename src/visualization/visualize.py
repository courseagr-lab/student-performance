import matplotlib.pyplot as plt
import seaborn as sns

from src.data.data_cleaning import project_path


def save_fig(fig, name, figures_path):
    figures_dir = project_path(figures_path)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", bbox_inches="tight", dpi=150)


def plot_numeric_distributions(df, numeric_cols, figures_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    save_fig(fig, "numeric_distributions", figures_path)
    plt.show()


def plot_target_balance(df, target, figures_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    df[target].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Distribusi target: {target}")
    save_fig(fig, "target_balance", figures_path)
    plt.show()


def plot_categorical_vs_target(df, cat_col, target, figures_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x=cat_col, y=target, ax=ax, errorbar=None)
    ax.set_title(f"{cat_col} vs {target} (rata-rata)")
    save_fig(fig, f"{cat_col}_vs_{target}", figures_path)
    plt.show()


def plot_correlation_heatmap(df, numeric_cols, target, figures_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[numeric_cols + [target]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    save_fig(fig, "correlation_heatmap", figures_path)
    plt.show()