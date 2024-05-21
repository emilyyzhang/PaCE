import pandas as pd
import numpy as np
from linearmodels import PanelOLS
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime


def apply_treatment(df, unit, time_period, 
                    treatment_name, fraction_treated, avg_treatment_length=None):
    
    # Ensure input fraction_treated is between 0 and 1
    if not (0 <= fraction_treated <= 1): 
        raise ValueError("fraction_treated must be between 0 and 1")
    
    # Create a new column for treatment, initialize with zeros
    df[treatment_name] = 0
    
    # Determine the number of units to treat
    unique_units = df[unit].unique()
    num_treated_units = int(np.ceil(fraction_treated * len(unique_units)))
    
    # Randomly select units to treat
    treated_units = np.random.choice(unique_units, num_treated_units, replace=False)
    
    # Sort the unique years to ensure chronological order
    sorted_years = np.sort(df[time_period].unique())

    if avg_treatment_length is None: avg_treatment_length = len(sorted_years) // 2 # this is average treatment length

    # Apply the treatment to each selected unit with a unique start year
    for unit_name in treated_units:
        # Randomly select a start year for this unit
        start_year = np.random.choice(sorted_years)
        actual_treatment_length = int(max(1, np.round(np.random.normal(avg_treatment_length, 1))))

        # Determine the treatment end year based on the start year and treatment length
        start_index = np.where(sorted_years == start_year)[0][0]
        end_index = min(start_index + actual_treatment_length, len(sorted_years))
        treatment_end_year = sorted_years[end_index - 1]  # Subtract 1 to include the start year in the count

        # Apply the treatment from the start year to the treatment end year for this unit
        df.loc[(df[unit] == unit_name) & (df[time_period] >= start_year) & (df[time_period] <= treatment_end_year), treatment_name] = 1

    return df


def apply_treatment_randomly(df, treatment_name, p):
    # Create a new column for the treatment, initialize to 0
    df[treatment_name] = 0
    
    # Assign the treatment with probability p to each row
    treatment_mask = np.random.rand(len(df)) < p
    df.loc[treatment_mask, treatment_name] = 1
    
    return df



def apply_treatment_adaptively(df, unit, time_period, outcome, treatment_name, fraction_treated):
    # Create a new column for treatment, initialize with zeros
    df[treatment_name] = 0
    
    # Get unique units and sort the time periods
    unique_units = df[unit].unique()
    sorted_years = np.sort(df[time_period].unique())

    # Create a dictionary to store percentage changes for each unit per time period
    percentage_changes = {year: {} for year in sorted_years}

    # Calculate percentage changes for each unit for each year, skipping the first two years
    for year_index in range(2, len(sorted_years)):
        this_year = sorted_years[year_index]
        prev_year = sorted_years[year_index - 1]
        prev_prev_year = sorted_years[year_index - 2]
        for unit_name in unique_units:
            prev_value = df.loc[(df[unit] == unit_name) & (df[time_period] == prev_prev_year), outcome].values
            current_value = df.loc[(df[unit] == unit_name) & (df[time_period] == prev_year), outcome].values
            if prev_value[0] != 0:
                percentage_change = abs((current_value[0] - prev_value[0]) / prev_value[0]) * 100
                percentage_changes[this_year][unit_name] = percentage_change
    
    # Treat units based on percentage change
    for year in sorted_years[2:]:  # start from the third year
        year_changes = percentage_changes[year]
        # Select top fraction_treated/2 units based on percentage change
        num_treated_units = int(np.ceil((fraction_treated / 2) * len(unique_units)))
        units_to_treat = sorted(year_changes, key=year_changes.get, reverse=True)[:num_treated_units]

        # Set treatment for selected units in this year
        for unit_name in units_to_treat:
            df.loc[(df[unit] == unit_name) & (df[time_period] == year), treatment_name] = 1

    return df




######################################################



def sigmoid(x):
    return 1 + 1 / (1 + np.exp(-20 * (x - 1/3)))

def apply_treatment_effect(df, treatment_name, covariates, OUTCOME,
                           function, effect_direction):
    # Sample a few (2-4) covariates
    num_covariates = 2
    sampled_covariates = np.random.choice(covariates, num_covariates, replace=False)
    
    # Initialize effect column
    effect_column_name = treatment_name + "_effect"
    
    if function == "additive":
        # Additive function: sum of the selected covariates
        df[effect_column_name] = df[sampled_covariates].sum(axis=1)
    elif function == "multiplicative":
        # Multiplicative function: product of the selected covariates
        df[effect_column_name] = df[sampled_covariates].prod(axis=1)
    elif function == "constant":
        # Constant function: set all values to a given constant
        df[effect_column_name] = 1
    elif function == "smooth":
        # Smooth function: sigmoid function of the first two features
        df[effect_column_name] = sigmoid(df[sampled_covariates[0]]) * sigmoid(df[sampled_covariates[1]])

    # Normalize the treatment effect so its mean is 5% of the mean of the OUTCOME column
    df[effect_column_name] *= 0.2 * df[OUTCOME].mean() / df[effect_column_name].mean()

    # Determine the direction of the effect
    if effect_direction == "negative": df[effect_column_name] = -df[effect_column_name]

    # Increment or decrement the OUTCOME column by the treatment effect where the treatment is applied
    df[OUTCOME] += df[effect_column_name] * df[treatment_name]

    return df



######################################################

## Estimators

def fixed_effect_treatment_estimator(df, unit_col, time_col, outcome_col, treatment_col, covariates):
    '''
    Implemented to return an average treatment effect, not heterogeneous
    '''
    covariates = [cov for cov in covariates if cov != outcome_col]

    # Set the index for panel data
    df.set_index([unit_col, time_col], inplace=True)
    
    # Prepare the model data
    X = df[[treatment_col] + covariates]
    Y = df[outcome_col]

    # Fit the Fixed Effects model with entity and time effects, and handle absorbed variables
    model = PanelOLS(Y, X, entity_effects=True, time_effects=True, drop_absorbed=True)
    results = model.fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    
    # The treatment effect is the coefficient of the treatment variable
    treatment_effect = results.params[treatment_col]

    # Reset index to bring back to the standard DataFrame format
    df.reset_index(inplace=True)
    
    return treatment_effect




def MCNNM(df, unit_col, time_col, outcome_col, treatment_col, suggest_r):

    def missing_algorithm(O, Ω, l, suggest = []):
        if (len(suggest) == 0):  M = np.zeros_like(O)
        else:  M = suggest[0]

        for _ in range(2000):
            u,s,vh = np.linalg.svd(O*Ω + M*(1-Ω), full_matrices = False)
            s = np.maximum(s - l, 0)
            M_new = (u*s).dot(vh)

            if (np.sum((M-M_new)**2) < 1e-12 * np.sum(M**2)): 
                break

            M = M_new
        return M, (1-Ω)*(O-M)

    def tune_missing_algorithm_with_rank(O, Ω, suggest_r):
        u, s, vh = np.linalg.svd(O, full_matrices = False)
        l = s[1]*1.1
            
        pre_M, tau = missing_algorithm(O, Ω, l)
        while (np.linalg.matrix_rank(pre_M) < suggest_r):
            l = l /1.5
            M, tau = missing_algorithm(O, Ω, l, suggest = [pre_M])            
            pre_M = M
        return tau


    O = df.pivot_table(index=unit_col, columns=time_col, values=outcome_col)
    Ω = 1-df.pivot_table(index=unit_col, columns=time_col, values=treatment_col).values

    tau = tune_missing_algorithm_with_rank(O.values, Ω, suggest_r=suggest_r)
    tau_df = pd.DataFrame(tau, index=O.index, columns=O.columns)
    tau_df = tau_df.stack().reset_index()
    tau_df.columns = [unit_col, time_col, 'MCMNN_result']

    df = df.merge(tau_df, on=[unit_col, time_col], how='left')
    return df['MCMNN_result']




############################################################################################################################################################





# Initialize the DataFrame to store scores
def initialize_scores():
    return pd.DataFrame(columns=[
        "Unit", "Adaptive Treatment", "Fraction Treated", "Treatment Effect Function", "Method", "NMAE", "NMAE Treated Only"
    ])

# Update metrics in the DataFrame
def update_metrics(df, unit, adaptive_treatment, fraction_treated, function, method, effects, true_effects, T):
    nmae = mean_absolute_error(true_effects, effects) / np.mean(np.abs(true_effects))
    nmae_treated_only = (mean_absolute_error(true_effects[T == 1], effects[T == 1]) / np.mean(np.abs(true_effects[T == 1]))) if T.any() else None
    if method == "MCNNM": nmae = 0

    new_row = pd.DataFrame({
        "Unit": [unit],
        "Adaptive Treatment": [adaptive_treatment],
        "Fraction Treated": [fraction_treated],
        "Treatment Effect Function": [function],
        "Method": [method],
        "NMAE": [nmae],
        "NMAE Treated Only": [nmae_treated_only]
    })
    return pd.concat([df, new_row], ignore_index=True)


# Save the DataFrame to CSV with a unique name
def save_results(df):
    unique_filename = f"results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(unique_filename, index=False)
    print(f"Results saved to {unique_filename}")




from econml.dml import LinearDML, CausalForestDML, DML
from econml.panel.dml import DynamicDML
from econml.dr import DRLearner
from econml.grf import CausalForest
from econml.dr import ForestDRLearner
from econml.metalearners import XLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from estimator import *


def run_experiment(df_orig, unit, time_period, outcome, covariates, effect_direction, parameters, num_iterations, suggest_r):
    
    scores_df = initialize_scores()

    for adaptive_treatment_application, fraction_treated, function in parameters:

        for _ in range(num_iterations):

            # Apply treatment and treatment effects with current parameters
            df = df_orig.copy()
            if adaptive_treatment_application: 
                df = apply_treatment_adaptively(df, unit, time_period, outcome, 'Treatment', fraction_treated)
            else: 
                df = apply_treatment(df, unit, time_period, 'Treatment', fraction_treated)
            df = apply_treatment_effect(df, 'Treatment', covariates, outcome, function, effect_direction)

            # Setting up the data
            X = df[covariates]
            # X_encoded = pd.get_dummies(df[unit], prefix=unit) 
            # X = pd.concat([X, X_encoded], axis=1)        
            T = df['Treatment'] # treatment mask
            Y = df[outcome]
            true_effects = df['Treatment_effect']

            estimator = TreatmentEffectEstimator(df.reset_index(drop=True).drop('Treatment_effect', axis=1), unit, time_period, outcome, covariates, ['Treatment'], suggest_r=suggest_r, splitting_criterion="MAE")
            estimator.fit(max_leaves = 40)
            scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'My Estimator', estimator.effect().values.flatten(), true_effects, T)

            effects = MCNNM(df, unit, time_period, outcome, 'Treatment', suggest_r = suggest_r)
            scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'MCNNM', effects, true_effects, T)

            models = {
                "LinearDML": LinearDML(model_y=RandomForestRegressor(), model_t=RandomForestClassifier(), discrete_treatment=True, categories=[0, 1]), 
                "DML": DML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingClassifier(), model_final=LassoCV(fit_intercept=False), discrete_treatment=True, categories=[0, 1]),
                "CausalForestDML": CausalForestDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingClassifier(), discrete_treatment=True, categories=[0, 1]),
                "DRLearner": DRLearner(),
                "ForestDRLearner": ForestDRLearner(model_regression=GradientBoostingRegressor(), model_propensity=GradientBoostingClassifier()),
                "XLearner": XLearner(models=GradientBoostingRegressor(), propensity_model=GradientBoostingClassifier())
            }
            for name, model in models.items():
                model.fit(Y, T, X=X)
                effects = model.effect(X).flatten()
                scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, name, effects, true_effects, T)

            # R integration using R's GRF package
            with localconverter(ro.default_converter + pandas2ri.converter):
                ro.globalenv['X'] = pandas2ri.py2rpy(X)
                ro.globalenv['T'] = ro.r('factor')(pandas2ri.py2rpy(T))
                ro.globalenv['Y'] = pandas2ri.py2rpy(Y)
            effects = ro.r('''library(grf)
                                    forest <- multi_arm_causal_forest(X, Y, W=T)
                                    predict(forest)$predictions  ''')
            effects = np.array(effects).flatten()
            scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'Causal Forest (R)', effects, true_effects, T)

    save_results(scores_df)
