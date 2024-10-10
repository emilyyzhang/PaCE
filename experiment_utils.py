import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime
import time

from econml.dml import LinearDML, CausalForestDML, DML
from econml.dr import DRLearner, ForestDRLearner
from econml.metalearners import XLearner, SLearner, TLearner, DomainAdaptationLearner
from econml.orf import DROrthoForest, DMLOrthoForest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from estimator import *
from causaltensor.cauest import DID
from causaltensor.cauest import MC_NNM_with_cross_validation
from causaltensor.cauest import DC_PR_auto_rank
from ganite import Ganite
import torch
from sklearn.preprocessing import StandardScaler
import uuid

import torch
import torch.nn as nn
import torch.optim as optim

# Global variable to hold the experiment ID for the current run
experiment_id = None

## Note to self: methods that are only for treatment effect of the treated entries: DID, DC-PR, MC-NNM, Mean Effect, Median Effect, KRLS, GANITE



def run_experiment(df_orig, unit, time_period, outcome, covariates, effect_direction, parameters, num_iterations, suggest_r):
    scores_df = initialize_scores(); global experiment_id

    for adaptive_treatment_application, fraction_treated, function in parameters:
        for _ in range(num_iterations):

            experiment_id = str(uuid.uuid4())  # Generates a random unique ID for the experiment

            # Apply treatment and treatment effects with current parameters
            df = df_orig.copy()
            if adaptive_treatment_application:
                df = apply_treatment_adaptively(df, unit, time_period, outcome, 'Treatment', fraction_treated)
            else:
                df = apply_treatment(df, unit, time_period, 'Treatment', fraction_treated)
            df = apply_treatment_effect(df, 'Treatment', covariates, outcome, function, effect_direction)

            # Setting up the data
            X = df[covariates]
            T = df['Treatment']
            Y = df[outcome]
            true_effects = df['Treatment_effect']

            # scores_df = benchmark_causaltensor(scores_df, df, unit, time_period, outcome, function, true_effects, T, adaptive_treatment_application, fraction_treated)
            # scores_df = benchmark_pace(scores_df, df, unit, time_period, outcome, covariates, suggest_r, true_effects, T, adaptive_treatment_application, fraction_treated, function)
            # scores_df = benchmark_econml(scores_df, X, Y, T, unit, adaptive_treatment_application, fraction_treated, function, true_effects)
            # scores_df = benchmark_mean_median_effects(scores_df, true_effects, T, unit, adaptive_treatment_application, fraction_treated, function)
            # scores_df = benchmark_grf_r(scores_df, X, T, Y, unit, adaptive_treatment_application, fraction_treated, function, true_effects)
            # scores_df = benchmark_krls_r(scores_df, X, T, Y, unit, adaptive_treatment_application, fraction_treated, function, true_effects)
            # scores_df = benchmark_nn_estimator(scores_df, X, T, Y, unit, adaptive_treatment_application, fraction_treated, function, true_effects)
            # leave out GANITE bc it is bad and slow
            # scores_df = benchmark_ganite(scores_df, df, unit, time_period, outcome, covariates, true_effects, T, adaptive_treatment_application, fraction_treated, function)  
            scores_df = benchmark_my_MCNNM(scores_df, df, unit, time_period, outcome, 'Treatment', true_effects, T, adaptive_treatment_application, fraction_treated, function, suggest_r)

    save_results(scores_df)

def benchmark_causaltensor(scores_df, df, unit, time_period, outcome, function, true_effects, T, adaptive_treatment_application, fraction_treated):
    # Compute O and Z
    O = df.pivot_table(index=unit, columns=time_period, values=outcome).values
    Z = df.pivot_table(index=unit, columns=time_period, values='Treatment').values

    # Benchmark 1: DID
    start_time = time.time()
    M, tau = DID(O, Z)
    effects = (Z * (O - M)).reshape(-1)
    time_taken = time.time() - start_time
    scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'DID', effects, true_effects, T, time_taken)

    # Benchmark 2: DC-PR
    start_time = time.time()
    M = DC_PR_auto_rank(O, Z)[0]
    effects = (Z * (O - M)).reshape(-1)
    time_taken = time.time() - start_time
    scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'DC-PR', effects, true_effects, T, time_taken)

    # # Benchmark 3: MC-NNM # Note: poor performance
    # start_time = time.time()
    # M, a, b, tau = MC_NNM_with_cross_validation(O, 1 - Z)
    # n1, n2 = O.shape
    # one_row, one_col = np.ones((1, n2)), np.ones((n1, 1))
    # M_adjusted = M + a.dot(one_row) + one_col.dot(b.T)
    # effects = (Z * (O - M_adjusted)).reshape(-1)
    # time_taken = time.time() - start_time
    # scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'MC-NNM', effects, true_effects, T, time_taken)

    return scores_df

def benchmark_my_MCNNM(scores_df, df, unit_col, time_col, outcome_col, treatment_col, true_effects, T, adaptive_treatment_application, fraction_treated, function, suggest_r):
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

    start_time = time.time()
    tau = tune_missing_algorithm_with_rank(O.values, Ω, suggest_r=suggest_r)
    time_taken = time.time() - start_time
    effects_flat = tau.reshape(-1)

    scores_df = update_metrics(scores_df, unit_col, adaptive_treatment_application, fraction_treated, function, 'MC-NNM (mine)', effects_flat, true_effects, T, time_taken)
    return scores_df



def benchmark_pace(scores_df, df, unit, time_period, outcome, covariates, suggest_r, true_effects, T, adaptive_treatment_application, fraction_treated, function):
    start_time = time.time(); global experiment_id
    estimator = TreatmentEffectEstimator(df.reset_index(drop=True).drop('Treatment_effect', axis=1), unit, time_period, outcome, covariates, 
                                         ['Treatment'], suggest_r=suggest_r, splitting_criterion="MAE", use_little_m=True)
    
    for max_leaves in range(1, 41):
        # Fit the estimator with the current max_leaves
        estimator.fit(max_leaves=max_leaves)

        # Determine the maximum number of leaves across all treatments
        max_leaves_across_treatments = max(len(estimator.leaves[treatment_name]) for treatment_name in estimator.columns_for_Z)

        effects = estimator.effect().values.flatten()
        time_taken = time.time() - start_time

        # Call update_metrics to store results in scores_df
        scores_df = update_metrics(
            df=scores_df,
            unit=unit,
            adaptive_treatment=adaptive_treatment_application,
            fraction_treated=fraction_treated,
            function=function,
            method=f"PaCE_{max_leaves_across_treatments}",  # This should be number of leaves instead of the max_leaves
            effects=effects,
            true_effects=true_effects,
            T=T,
            time_taken=time_taken
        )

    # Call update_metrics to store results in scores_df
    scores_df = update_metrics(
        df=scores_df,
        unit=unit,
        adaptive_treatment=adaptive_treatment_application,
        fraction_treated=fraction_treated,
        function=function,
        method=f"PaCE",  
        effects=effects,
        true_effects=true_effects,
        T=T,
        time_taken=time_taken
    )

    return scores_df

def benchmark_econml(scores_df, X, Y, T, unit, adaptive_treatment_application, fraction_treated, function, true_effects):
    models = {
        "LinearDML": LinearDML(model_y=RandomForestRegressor(), model_t=RandomForestClassifier(), discrete_treatment=True, categories=[0, 1]),
        "DML": DML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingClassifier(), model_final=LassoCV(fit_intercept=False), discrete_treatment=True, categories=[0, 1]),
        "CausalForestDML": CausalForestDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingClassifier(), discrete_treatment=True, categories=[0, 1]),
        "DRLearner": DRLearner(),
        "ForestDRLearner": ForestDRLearner(model_regression=GradientBoostingRegressor(), model_propensity=GradientBoostingClassifier()),
        "XLearner": XLearner(models=GradientBoostingRegressor(), propensity_model=GradientBoostingClassifier()),
        "DomainAdaptationLearner": DomainAdaptationLearner(models=RandomForestRegressor(), final_models=RandomForestRegressor()),
        "SLearner": SLearner(overall_model=RandomForestRegressor()),
        "TLearner": TLearner(models=RandomForestRegressor()),
        # "DROrthoForest": DROrthoForest(n_trees=500, min_leaf_size=10, max_depth=10, subsample_ratio=1.0), # warning
        # "DMLOrthoForest": DMLOrthoForest(n_trees=500, min_leaf_size=10, max_depth=10, model_Y=RandomForestRegressor(), model_T=RandomForestClassifier()) #slow
    }
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(Y, T, X=X)
        effects = model.effect(X).flatten()
        time_taken = time.time() - start_time
        scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, name, effects, true_effects, T, time_taken)
    return scores_df

def benchmark_mean_median_effects(scores_df, true_effects, T, unit, adaptive_treatment_application, fraction_treated, function):
    start_time = time.time()
    mean_effect = np.mean(true_effects)
    effects = np.full(true_effects.shape, mean_effect)
    time_taken = time.time() - start_time
    scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'Mean Effect', effects, true_effects, T, time_taken)

    start_time = time.time()
    median_effect = np.median(true_effects)
    effects = np.full(true_effects.shape, median_effect)
    time_taken = time.time() - start_time
    scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'Median Effect', effects, true_effects, T, time_taken)

    return scores_df

def benchmark_grf_r(scores_df, X, T, Y, unit, adaptive_treatment_application, fraction_treated, function, true_effects):
    start_time = time.time()
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['X'] = pandas2ri.py2rpy(X)
        ro.globalenv['T'] = ro.r('factor')(pandas2ri.py2rpy(T))
        ro.globalenv['Y'] = pandas2ri.py2rpy(Y)
    effects = ro.r('''library(grf)
                        forest <- multi_arm_causal_forest(X, Y, W=T)
                        predict(forest)$predictions  ''')
    effects = np.array(effects).flatten()
    time_taken = time.time() - start_time
    scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'Causal Forest (R)', effects, true_effects, T, time_taken)
    return scores_df

def benchmark_krls_r(scores_df, X, T, Y, unit, adaptive_treatment_application, fraction_treated, function, true_effects):
    # only run for State dataset, as it does not scale for the larger datasets
    if unit !='State Name': return scores_df
    
    # Prepare the untreated entries
    untreated_X = X[T == 0]
    untreated_Y = Y[T == 0]
    
    start_time = time.time()
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv['untreated_X'] = pandas2ri.py2rpy(untreated_X)
        ro.globalenv['untreated_Y'] = pandas2ri.py2rpy(untreated_Y)
        ro.globalenv['X'] = pandas2ri.py2rpy(X)
    
    # Train KRLS and make predictions
    ro.r('''
        suppressWarnings(suppressMessages({
            library(KRLS)
            krls_model <- krls(X=as.matrix(untreated_X), y=untreated_Y)
            predictions <- predict(krls_model, newdata=as.matrix(X))$fit
        }))
    ''')
    counterfactuals = np.array(ro.globalenv['predictions']).flatten()
    time_taken = time.time() - start_time
    
    # Calculate treatment effects
    treatment_effects = Y - counterfactuals

    scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'KRLS (R)', treatment_effects, true_effects, T, time_taken)
    return scores_df

def benchmark_ganite(scores_df, df, unit, time_period, outcome, covariates, true_effects, T, adaptive_treatment_application, fraction_treated, function):
    # only run for State dataset, as it does not scale for the larger datasets
    if unit !='State Name': return scores_df
    
    # Prepare data for GANITE
    X = df[covariates].values
    Y = df[outcome].values.reshape(-1, 1)
    T_reshaped = df['Treatment'].values.reshape(-1, 1)
    
    start_time = time.time()
    
    # Initialize and train GANITE model
    ganite_model = Ganite(X, T_reshaped, Y, num_iterations=500)
    pred = ganite_model(X).numpy().flatten()

    effects =  Y.flatten() - pred
    time_taken = time.time() - start_time
    scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'GANITE', effects, true_effects, T, time_taken)
    return scores_df

def benchmark_nn_estimator(scores_df, X, T, Y, unit, adaptive_treatment_application, fraction_treated, function, true_effects):
    # only run for State dataset, as it does not scale for the larger datasets
    if unit !='State Name': return scores_df

    class NNEstimator(nn.Module):
        def __init__(self, input_dim, hidden_layers, dropout_rate):
            super(NNEstimator, self).__init__()
            layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim
            self.feature_extractor = nn.Sequential(*layers)
            self.treated_outcome = nn.Linear(prev_dim, 1)
            self.control_outcome = nn.Linear(prev_dim, 1)

        def forward(self, x, t):
            phi = self.feature_extractor(x)
            y0_pred = self.control_outcome(phi)
            y1_pred = self.treated_outcome(phi)
            y_pred = t * y1_pred + (1 - t) * y0_pred
            return y_pred, y0_pred, y1_pred

        def train_model(self, X, T, Y, num_epochs=500, batch_size=64, learning_rate=1e-3):
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            dataset = torch.utils.data.TensorDataset(X, T, Y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            self.train()
            for epoch in range(num_epochs):
                for batch_X, batch_T, batch_Y in dataloader:
                    optimizer.zero_grad()
                    y_pred, y0_pred, y1_pred = self(batch_X, batch_T) # calls forward
                    loss = criterion(y_pred, batch_Y)
                    loss.backward()
                    optimizer.step()

        def predict_ite(self, X):
            self.eval()
            with torch.no_grad():
                _, y0_pred, y1_pred = self(X, torch.zeros_like(X[:, 0]))
                _, y0_pred, y1_pred = self(X, torch.ones_like(X[:, 0]))
            return y1_pred - y0_pred

    start_time = time.time()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert data to torch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    T_tensor = torch.tensor(T.values, dtype=torch.float32).unsqueeze(1)
    Y_tensor = torch.tensor(Y.values, dtype=torch.float32).unsqueeze(1)

    # Initialize and train the NNEstimator model
    nn_estimator = NNEstimator(input_dim=X.shape[1], hidden_layers=[64, 32], dropout_rate=0.5)
    nn_estimator.train_model(X_tensor, T_tensor, Y_tensor, num_epochs=500, batch_size=64, learning_rate=1e-3)

    # Predict treatment effects
    treatment_effects = nn_estimator.predict_ite(X_tensor).detach().numpy().flatten()
    
    time_taken = time.time() - start_time

    # Update the scores dataframe
    scores_df = update_metrics(scores_df, unit, adaptive_treatment_application, fraction_treated, function, 'NNEstimator', treatment_effects, true_effects, T, time_taken)
    return scores_df







############################################################################################################################################################




# Initialize the DataFrame to store scores
def initialize_scores():
    return pd.DataFrame(columns=[
        "Unit", "Adaptive Treatment", "Fraction Treated", "Treatment Effect Function", "Method", "NMAE", "NMAE Treated Only", 
        "MSE", "MSE Treated Only", "Time Taken"
    ])

# Update metrics in the DataFrame
def update_metrics(df, unit, adaptive_treatment, fraction_treated, function, method, effects, true_effects, T, time_taken):
    global experiment_id

    nmae = mean_absolute_error(true_effects, effects) / np.mean(np.abs(true_effects))
    nmae_treated_only = (mean_absolute_error(true_effects[T == 1], effects[T == 1]) / np.mean(np.abs(true_effects[T == 1]))) if np.any(T) else None

    mse = mean_squared_error(true_effects, effects)
    mse_treated_only = mean_squared_error(true_effects[T == 1], effects[T == 1]) if np.any(T) else None
    # if method == "MCNNM": nmae = 0

    new_row = pd.DataFrame({
        "Experiment ID": [experiment_id],  
        "Unit": [unit],
        "Adaptive Treatment": [adaptive_treatment],
        "Fraction Treated": [fraction_treated],
        "Treatment Effect Function": [function],
        "Method": [method],
        "NMAE": [nmae],
        "NMAE Treated Only": [nmae_treated_only],
        "MSE": [mse],
        "MSE Treated Only": [mse_treated_only],
        "Time Taken": [time_taken],
        "Estimated ATE": [np.mean(effects)], 
        "True ATE": [np.mean(true_effects)],
        "Estimated ATT": [np.mean(effects[T == 1])],
        "True ATT": [np.mean(true_effects[T == 1])]
    })

    return pd.concat([df, new_row], ignore_index=True)

# Save the DataFrame to CSV with a unique name
def save_results(df):
    unique_filename = f"results/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(unique_filename, index=False)
    print(f"Results saved to {unique_filename}")


############################################################################################################################################################





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


