""" Propensity Score Matching Class"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from typing import List
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def plot_cdf(
    df_muestra,
    df_referencia,
    column,
    labels=("Muestra", "Referencia Global"),
    colors=("blue", "red"),
):
    """
    Plot CDFs for a specified column from two DataFrames on the same plot with mean and sd annotations.

    Parameters:
    df_muestra (pd.DataFrame): First DataFrame.
    df_referencia (pd.DataFrame): Second DataFrame.
    column (str): Column name to plot the CDF for.
    labels (tuple): Labels for the DataFrames.
    colors (tuple): Colors for the CDFs of each DataFrame.
    """
    plt.figure(figsize=(8, 6))

    # Calculate and plot CDF for the first DataFrame
    data1 = df_muestra[column].dropna().sort_values()
    cdf1 = np.arange(len(data1)) / float(len(data1) - 1)
    plt.plot(
        data1,
        cdf1,
        label=f"{labels[0]} - Mean: {data1.mean():.2f},\nSD: {data1.std():.2f}",
        color=colors[0],
    )

    # Calculate and plot CDF for the second DataFrame
    data2 = df_referencia[column].dropna().sort_values()
    cdf2 = np.arange(len(data2)) / float(len(data2) - 1)
    plt.plot(
        data2,
        cdf2,
        label=f"{labels[1]} - Mean: {data2.mean():.2f},\nSD: {data2.std():.2f}",
        color=colors[1],
    )

    plt.title(f"Cumulative Distribution Function of {column}")
    plt.xlabel(column)
    plt.ylabel("CDF")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_histogram(
    data: pd.DataFrame,
    treatment_col: str,
    value_col: str,
    titulo: str,
    plot_size: tuple = (10, 6),
):
    """
    Plot the propensity scores for the treatment and control groups.
    """
    fig, ax = plt.subplots(figsize=plot_size)

    # Visualize propensity scores
    sns.kdeplot(
        data=data[data[treatment_col] == True],
        x=value_col,
        fill=True,
        color="#5af8bd",
        label="Tratamiento",
        ax=ax,
    )
    sns.kdeplot(
        data=data[data[treatment_col] == False],
        x=value_col,
        fill=True,
        color="#131538",
        label="Control",
        ax=ax,
    )
    ax.set_title(titulo)
    ax.legend()

    plt.tight_layout()
    plt.show()


class PSMatch:
    def __init__(
        self, data: pd.DataFrame, treatment_col: str, outcome_col: str, id_col: str
    ):
        """
        Initialize the PSMatch class.
        Parameters:
            data: pd.DataFrame: The data to be used for propensity score matching.
            treatment_col: str: The column name of the treatment variable used to identify the treated and control groups.
            outcome_col: str: The column name of the variable used to evaluate the effect of the treatment.
        """
        self.data = data
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.id_col = id_col
        self.propensity_scores = None
        self.matched_data = None

    def estimate_propensity_scores(
        self, numerical_covariates: List[str], categorical_covariates: List[str] = None
    ):
        """
        Estimate propensity scores using logistic regression.
        Parameters:
            numerical_covariates: list: The list of numerical covariates to be scaled.
            categorical_covariates: list: The list of categorical covariates to be encoded.
        """
        # Define preprocessing for numerical and categorical columns
        if categorical_covariates is not None:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numerical_covariates),
                    ("cat", OneHotEncoder(), categorical_covariates),
                ]
            )
        else:
            preprocessor = ColumnTransformer(
                transformers=[("num", StandardScaler(), numerical_covariates)]
            )

        # Define the logistic regression model
        self.model = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression())]
        )

        # Separate features and target
        if categorical_covariates is None:
            X = self.data[numerical_covariates]
        else:
            X = self.data[numerical_covariates + categorical_covariates]
        y = self.data[self.treatment_col]

        # Impute missing values
        X = X.fillna(X.mean())

        # Fit the model
        self.model.fit(X, y)

        # Predict propensity scores
        self.data["propensity_score"] = self.model.predict_proba(X)[:, 1]
        self.propensity_scores = self.data["propensity_score"]

        return self.data[["propensity_score"]]

    def plot_propensity_scores(self):
        """
        Plot the propensity scores for the treatment and control groups.
        """
        fig, ax = plt.subplots(figsize=(7, 4))

        # Visualize propensity scores
        sns.kdeplot(
            data=self.data[self.data[self.treatment_col] == 0],
            x="propensity_score",
            fill=True,
            color="#5af8bd",
            label="Control",
            ax=ax,
        )
        sns.kdeplot(
            data=self.data[self.data[self.treatment_col] == 1],
            x="propensity_score",
            fill=True,
            color="#131538",
            label="Treated",
            ax=ax,
        )
        ax.set_title("Propensity")
        ax.legend()

        plt.tight_layout()
        plt.show()

    def match(self, k=1, with_replacement=True):
        """
        Match individuals in the treatment group to k individuals in the control group
        based on the propensity scores.
        Parameters:
            k: int: The number of control individuals to match to each treated individual.
            with_replacement: bool: If True, match with replacement. If False, match without replacement.
        """
        # Separate treated and control groups
        treated = self.data[self.data[self.treatment_col] == 1].copy()
        control = self.data[self.data[self.treatment_col] == 0].copy()

        # Create a NearestNeighbors model for finding nearest neighbors
        nn = NearestNeighbors(n_neighbors=k)

        matched_pairs = []
        used_control_indices = set()

        for treated_idx in treated.index:
            # Fit nearest neighbors excluding already used control units if without replacement
            if with_replacement:
                available_control = control
            else:
                available_control = control.loc[
                    ~control.index.isin(used_control_indices)
                ]

            # Adjust k if there are fewer available control units than k
            current_k = min(k, len(available_control))
            if current_k == 0:
                break  # No more control units available for matching

            nn.set_params(n_neighbors=current_k)
            nn.fit(available_control[["propensity_score"]])

            # Find nearest neighbors for the treated unit
            distances, indices = nn.kneighbors(
                treated.loc[[treated_idx], ["propensity_score"]]
            )

            # Select the closest control units and ensure no replacement if specified
            for neighbor_idx in indices[0]:
                control_idx = available_control.index[neighbor_idx]
                matched_pairs.append(
                    {
                        "treated_id": treated.loc[treated_idx, self.id_col],
                        "control_id": control.loc[control_idx, self.id_col],
                        "treated_propensity": treated.loc[
                            treated_idx, "propensity_score"
                        ],
                        "control_propensity": control.loc[
                            control_idx, "propensity_score"
                        ],
                        "treated_outcome": treated.loc[treated_idx, self.outcome_col],
                        "control_outcome": control.loc[control_idx, self.outcome_col],
                    }
                )
                if not with_replacement:
                    used_control_indices.add(control_idx)
                    if len(used_control_indices) == len(control):
                        break  # All control units have been used

        self.matched_data = pd.DataFrame(matched_pairs)

        return self.matched_data

    def plot_cumulative_distribution(
        self, matched_data: pd.DataFrame, historic_data: pd.DataFrame
    ):
        """ """
        control_ids = list(matched_data["control_id"].unique())
        treatmen_id = list(matched_data["treated_id"].unique())

        start_date = self.data[self.data["TRATAMIENTO_INICIO"] == True]["PERIODO"].max()
        latest_date = self.data["PERIODO"].max()

        historic_data[self.outcome_col] = pd.to_numeric(
            historic_data[self.outcome_col], errors="coerce"
        )

        control_pre = historic_data[
            (historic_data["CLIENTE"].isin(control_ids))
            & (historic_data["PERIODO"] <= start_date)
        ].copy()
        treatmen_pre = historic_data[
            (historic_data["CLIENTE"].isin(treatmen_id))
            & (historic_data["PERIODO"] <= start_date)
        ].copy()

        control_post = historic_data[
            (historic_data["CLIENTE"].isin(control_ids))
            & (historic_data["PERIODO"] > start_date)
        ].copy()
        treatmen_post = historic_data[
            (historic_data["CLIENTE"].isin(treatmen_id))
            & (historic_data["PERIODO"] > start_date)
        ].copy()

        plot_cdf(
            treatmen_pre,
            control_pre,
            self.outcome_col,
            labels=("Tratamiento_PRE", "Control_PRE"),
        )

        plot_cdf(
            treatmen_post,
            control_post,
            self.outcome_col,
            labels=("Tratamiento_POST", "Control_POST"),
        )

    def plot_sales(self, matched_data: pd.DataFrame, historic_data: pd.DataFrame):
        """ """
        control_ids = list(matched_data["control_id"].unique())
        treatmen_id = list(matched_data["treated_id"].unique())

        start_date = historic_data[historic_data["TRATAMIENTO_INICIO"] == True][
            "PERIODO"
        ].max()
        latest_date = historic_data["PERIODO"].max()

        # Two times Number of days between the start of the treatment and the end of the data
        treatment_days = (latest_date - start_date).days

        # Start date minus two times treatment_days
        limit_date = start_date - datetime.timedelta(days=2 * treatment_days)
        print(start_date, latest_date, limit_date)

        sales_evol = historic_data[
            (historic_data["CLIENTE"].isin(treatmen_id + control_ids))
        ][["CLIENTE", "GRUPO_TRATAMIENTO", "PERIODO", "TOTAL_SUBGRUPO"]]

        sales_evol["PERIODO"] = pd.to_datetime(sales_evol["PERIODO"])

        # Calculate the ventas_periodo DataFrame
        sales_divide = (
            sales_evol.groupby(["PERIODO", "GRUPO_TRATAMIENTO"])
            .agg(PROMEDIO_VENTAS_TIENDA=("TOTAL_SUBGRUPO", "mean"))
            .reset_index()
        )

        # Create the plot
        fig, ax = plt.subplots()
        sales_divide[sales_divide["PERIODO"] >= str(limit_date)].pivot_table(
            index="PERIODO",
            columns="GRUPO_TRATAMIENTO",
            values="PROMEDIO_VENTAS_TIENDA",
        ).plot(ax=ax)

        # Add a vertical line at start of trial
        ax.axvline(pd.to_datetime(start_date), color="red", linestyle="--", linewidth=2)

        # Adjust plot size
        fig.set_size_inches(14, 7)

        # Show the plot
        plt.show()

    def plot_sales_histogram(
        self, matched_data: pd.DataFrame, historic_data: pd.DataFrame
    ):
        """ """
        control_ids = list(matched_data["control_id"].unique())
        treatmen_id = list(matched_data["treated_id"].unique())

        start_date = historic_data[historic_data["TRATAMIENTO_INICIO"] == True][
            "PERIODO"
        ].max()

        sales_evol = historic_data[
            (historic_data["CLIENTE"].isin(treatmen_id + control_ids))
        ][["CLIENTE", "GRUPO_TRATAMIENTO", "PERIODO", "TOTAL_SUBGRUPO"]]

        # Calculate the ventas_periodo DataFrame
        sales_divide = (
            sales_evol.groupby(["PERIODO", "GRUPO_TRATAMIENTO"])
            .agg(PROMEDIO_VENTAS_TIENDA=("TOTAL_SUBGRUPO", "mean"))
            .reset_index()
        )

        plot_histogram(
            sales_divide[sales_divide["PERIODO"] <= start_date],
            "GRUPO_TRATAMIENTO",
            "PROMEDIO_VENTAS_TIENDA",
            "Distribución de Ventas por Dia Pre Piloto",
        )
        plot_histogram(
            sales_divide[sales_divide["PERIODO"] > start_date],
            "GRUPO_TRATAMIENTO",
            "PROMEDIO_VENTAS_TIENDA",
            "Distribución de Ventas por Dia Pre Piloto",
        )
