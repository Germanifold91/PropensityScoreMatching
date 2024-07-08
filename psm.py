""" Propensity Score Matching Class"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import pickle
from typing import List
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class PSMatch:
    def __init__(self, data: pd.DataFrame, treatment_col: str, outcome_col: str, id_col: str):
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
        
    def estimate_propensity_scores(self, numerical_covariates: List[str], categorical_covariates: List[str]=None):
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
                    ('num', StandardScaler(), numerical_covariates),
                    ('cat', OneHotEncoder(), categorical_covariates)
                ])
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_covariates)
                ])
        
        # Define the logistic regression model
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])
        
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
        self.data['propensity_score'] = self.model.predict_proba(X)[:, 1]
        self.propensity_scores = self.data['propensity_score']

        return self.data[['propensity_score']]
    
    def plot_propensity_scores(self):
        """
        Plot the propensity scores for the treatment and control groups.
        """
        fig, ax = plt.subplots(figsize=(7, 4))

        # Visualize propensity scores
        sns.kdeplot(data=self.data[self.data[self.treatment_col] == 0], x='propensity_score', fill=True, 
                    color='#5af8bd', label='Control', ax=ax)
        sns.kdeplot(data=self.data[self.data[self.treatment_col] == 1], x='propensity_score', fill=True, 
                    color='#131538', label='Treated', ax=ax)
        ax.set_title('Propensity')
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
                    available_control = control.loc[~control.index.isin(used_control_indices)]
                
                # Adjust k if there are fewer available control units than k
                current_k = min(k, len(available_control))
                if current_k == 0:
                    break  # No more control units available for matching
                
                nn.set_params(n_neighbors=current_k)
                nn.fit(available_control[['propensity_score']])
                
                # Find nearest neighbors for the treated unit
                distances, indices = nn.kneighbors(treated.loc[[treated_idx], ['propensity_score']])
                
                # Select the closest control units and ensure no replacement if specified
                for neighbor_idx in indices[0]:
                    control_idx = available_control.index[neighbor_idx]
                    matched_pairs.append({
                        'treated_id': treated.loc[treated_idx, self.id_col],
                        'control_id': control.loc[control_idx, self.id_col],
                        'treated_propensity': treated.loc[treated_idx, 'propensity_score'],
                        'control_propensity': control.loc[control_idx, 'propensity_score'],
                        'treated_outcome': treated.loc[treated_idx, self.outcome_col],
                        'control_outcome': control.loc[control_idx, self.outcome_col]
                    })
                    if not with_replacement:
                        used_control_indices.add(control_idx)
                        if len(used_control_indices) == len(control):
                            break  # All control units have been used
            
            self.matched_data = pd.DataFrame(matched_pairs)
            
            return self.matched_data