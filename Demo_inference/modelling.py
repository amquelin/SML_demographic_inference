import sys
import os
import math
import summary_statistics as ss
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
sys.path.append('./Simulations')
import default_settings as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt  
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.inspection import permutation_importance
import seaborn as sns



T_split_r = [min(st.demo_settings['T_split']), max(st.demo_settings['T_split'])]
mig_r = [min(st.demo_settings['migration_rate']), max(st.demo_settings['migration_rate'])]
N_anc_r = [min(st.demo_settings['N_ancestral']), max(st.demo_settings['N_ancestral'])]


limit_mapping = {
    'T_split': T_split_r,
    'migration_rate': mig_r,
    'N_ancestral': N_anc_r
}

def get_limit(Y):
    limit = limit_mapping.get(Y)
    if limit is None:
        raise ValueError("Invalid value for Y")
    return limit


def preprocess_data(df, var_tbr , Y , train_size):
    
    predictors = [col for col in df.columns if col not in var_tbr] # Remove no predictors columns from df
    
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0)
    df_train, df_test = train_test_split(df, test_size = 1 - train_size, random_state=11)

    print("Number of rows in the training set:", len(df_train))
    print("Number of rows in the test set:", len(df_test))
    
    # StandardScaler object for features
    x_scaler = StandardScaler()

    X_train_std = x_scaler.fit_transform(df_train[predictors])
    X_test_std = x_scaler.transform(df_test[predictors])

    X_train_std = pd.DataFrame(X_train_std, columns=predictors)
    X_test_std = pd.DataFrame(X_test_std, columns=predictors)

    X_train = df_train[predictors]
    X_test = df_test[predictors]   

    # StandardScaler for the response variable
    y_scaler = StandardScaler()

    Y_train_std = y_scaler.fit_transform(df_train[[Y]])
    Y_test_std = y_scaler.transform(df_test[[Y]])

    Y_train_std = pd.Series(Y_train_std.flatten(), name=Y)
    Y_test_std = pd.Series(Y_test_std.flatten(), name=Y)

    Y_train = df_train[Y]
    Y_test = df_test[Y] 
    
    var_tbr_data_test = df_test[var_tbr].copy()
    
    return X_train, X_test, Y_train, Y_test, X_train_std, X_test_std, Y_train_std, Y_test_std, var_tbr_data_test, y_scaler
    



def get_rf_params(Y):
    # Define the mapping from Y values to filenames
    file_mapping = {
        "T_split": "rf_param_T_split.json",
        "N_ancestral": "rf_param_N_ancestral.json",
        "migration_rate": "rf_param_migration_rate.json"
    }

    # Get the filename based on Y
    filename = file_mapping.get(Y)
    
    if filename is None:
        print(f"No file found for Y={Y}")
        return None
    
    # Define the directory containing the JSON files
    directory = "Demo_inference/modelling_parameters"
    
    # Create the full path to the file
    filepath = os.path.join(directory, filename)
    
    try:
        # Load the JSON file
        with open(filepath, 'r') as file:
            rf_params = json.load(file)

        # Extract the desired parameters
        desired_params = {
            "n_estimators": rf_params.get("n_estimators"),
            "max_depth": rf_params.get("max_depth"),
            "min_samples_split": rf_params.get("min_samples_split"),
            "min_samples_leaf": rf_params.get("min_samples_leaf"),
            "max_features": rf_params.get("max_features")
        }

        # Print the extracted parameters
        print(json.dumps(desired_params, indent=4))
        return desired_params

    except FileNotFoundError:
        print(f"File {filepath} not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {filepath}.")
        return rf_params
        
        
def get_xgb_params(Y):
    # Define the mapping from Y values to filenames
    file_mapping = {
        "T_split": "xgb_param_T_split.json",
        "N_ancestral": "xgb_param_N_ancestral.json",
        "migration_rate": "xgb_param_migration_rate.json"
    }

    # Get the filename based on Y
    filename = file_mapping.get(Y)
    
    if filename is None:
        print(f"No file found for Y={Y}")
        return None
    
    # Define the directory containing the JSON files
    directory = "Demo_inference/modelling_parameters"
    
    # Create the full path to the file
    filepath = os.path.join(directory, filename)
    
    try:
        # Load the JSON file
        with open(filepath, 'r') as file:
            xgb_params = json.load(file)

        # Extract the desired parameters
        desired_params = {
            "n_estimators": xgb_params.get("n_estimators"),
            "max_depth": xgb_params.get("max_depth"),
            "subsample": xgb_params.get("subsample"),
            "colsample_bytree": xgb_params.get("colsample_bytree")
        }

        # Print the extracted parameters
        print(json.dumps(desired_params, indent=4))
        return xgb_params

    except FileNotFoundError:
        print(f"File {filepath} not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {filepath}.")
        return None
        
        
def get_mlp_params(Y):
    # Define the mapping from Y values to filenames
    file_mapping = {
        "T_split": "mlp_param_T_split.json",
        "N_ancestral": "mlp_param_N_ancestral.json",
        "migration_rate": "mlp_param_migration_rate.json"
    }

    # Get the filename based on Y
    filename = file_mapping.get(Y)
    
    if filename is None:
        print(f"No file found for Y={Y}")
        return None
    
    # Define the directory containing the JSON files
    directory = "Demo_inference/modelling_parameters"
    
    # Create the full path to the file
    filepath = os.path.join(directory, filename)
    
    try:
        # Load the JSON file
        with open(filepath, 'r') as file:
            mlp_params = json.load(file)

        desired_params = {
            "hidden_layer_sizes": mlp_params.get("hidden_layer_sizes"),
            "activation": mlp_params.get("activation"),
            "solver": mlp_params.get("solver"),
            "batch_size": mlp_params.get("batch_size"),
            "max_iter": mlp_params.get("max_iter")
        }

        # Print the extracted parameters
        print(json.dumps(desired_params, indent=4))
        return mlp_params

    except FileNotFoundError:
        print(f"File {filepath} not found.")
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file {filepath}.")
        return None
    
def train_and_evaluate_rf(X_train, Y_train, X_test, Y_test, rf_params, Y, y_scaler=None, origin_r=False, limit=None):

    Y_min = limit[0]
    Y_max = limit[1]

    rf = RandomForestRegressor(**rf_params)
    rf.fit(X_train, Y_train)
    test_pred = rf.predict(X_test)


    if y_scaler and origin_r:
        test_pred_original = y_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        Y_test_original = y_scaler.inverse_transform(np.array(Y_test).reshape(-1, 1)).flatten()

        if limit is not None:
            lower_limit, upper_limit = limit
            test_pred_original = np.clip(test_pred_original, lower_limit, upper_limit)
        
        Y_test = Y_test_original   
        test_pred_rf = test_pred_original    
    
    # computation of metrics
    test_mse = mean_squared_error(Y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(Y_test, test_pred)
    test_nmae = test_mae / (Y_max - Y_min)
    r_squared = r2_score(Y_test, test_pred)
    
    # Print test MSE, RMSE, MAE, and R-squared for the model
    print("Test MSE :", test_mse)
    print("Test RMSE :", test_rmse)
    print("Test MAE:", test_mae)
    print("Test NMAE:", test_nmae)
    #print("R-squared for Random Forest:", r_squared)

    result_pred = pd.DataFrame({
        "Observed_target": Y_test,  # Observed values
        "Predicted_target": test_pred  # Predicted values
    })
    
#    return rf, test_mse, test_rmse, test_mae, test_nmae, r_squared, result_pred
    return rf, result_pred    
    
def train_and_evaluate_xgb(X_train, Y_train, X_test, Y_test, xgb_params, Y, y_scaler=None, origin_r=False, limit=None):

    Y_min = limit[0]
    Y_max = limit[1]

    xgb_r = xgb.XGBRegressor(**xgb_params)
    xgb_r.fit(X_train, Y_train)
    test_pred = xgb_r.predict(X_test)

    if y_scaler and origin_r:
        test_pred_original = y_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        Y_test_original = y_scaler.inverse_transform(np.array(Y_test).reshape(-1, 1)).flatten()

        if limit is not None:
            lower_limit, upper_limit = limit
            test_pred_original = np.clip(test_pred_original, lower_limit, upper_limit)
        
        Y_test = Y_test_original   
        test_pred = test_pred_original           
    
    # metrics
    test_mse = mean_squared_error(Y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(Y_test, test_pred)
    test_nmae = test_mae / (Y_max - Y_min)
    r_squared = r2_score(Y_test, test_pred)
    
    # Print test MSE, RMSE, MAE, and R-squared for the model
    print("Test MSE :", test_mse)
    print("Test RMSE :", test_rmse)
    print("Test MAE:", test_mae)
    print("Test NMAE:", test_nmae)
    #print("R-squared for Random Forest:", r_squared)

    result_pred = pd.DataFrame({
        "Observed_target": Y_test,  # Observed values
        "Predicted_target": test_pred  # Predicted values
    })    
    
#    return xgb, test_mse, test_rmse, test_mae, test_nmae, r_squared, result_pred
    return xgb_r, result_pred     
    
def train_and_evaluate_mlp(X_train, Y_train, X_test, Y_test, mlp_params, Y, y_scaler=None, origin_r=False, limit=None):
   
    # Convert the parameter values to appropriate types
    mlp_params['hidden_layer_sizes'] = tuple(mlp_params['hidden_layer_sizes'])
    mlp_params['alpha'] = float(mlp_params['alpha'])
    mlp_params['learning_rate_init'] = float(mlp_params['learning_rate_init'])
    mlp_params['power_t'] = float(mlp_params['power_t'])
    mlp_params['max_iter'] = int(mlp_params['max_iter'])
    mlp_params['shuffle'] = bool(mlp_params['shuffle'])
    mlp_params['random_state'] = None if mlp_params['random_state'] is None else int(mlp_params['random_state'])
    mlp_params['tol'] = float(mlp_params['tol'])
    mlp_params['verbose'] = bool(mlp_params['verbose'])
    mlp_params['momentum'] = float(mlp_params['momentum'])
    mlp_params['nesterovs_momentum'] = bool(mlp_params['nesterovs_momentum'])
    mlp_params['early_stopping'] = bool(mlp_params['early_stopping'])
    mlp_params['validation_fraction'] = float(mlp_params['validation_fraction'])
    mlp_params['beta_1'] = float(mlp_params['beta_1'])
    mlp_params['beta_2'] = float(mlp_params['beta_2'])
    mlp_params['epsilon'] = float(mlp_params['epsilon'])
    mlp_params['n_iter_no_change'] = int(mlp_params['n_iter_no_change'])
    mlp_params['max_fun'] = int(mlp_params['max_fun'])
    
    Y_min = limit[0]
    Y_max = limit[1]
    
    mlp = MLPRegressor(**mlp_params)
    mlp.fit(X_train, Y_train)
    test_pred = mlp.predict(X_test)

    if y_scaler and origin_r:
        test_pred_original = y_scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
        Y_test_original = y_scaler.inverse_transform(np.array(Y_test).reshape(-1, 1)).flatten()
        
        if limit is not None:
            lower_limit, upper_limit = limit
            test_pred_original = np.clip(test_pred_original, lower_limit, upper_limit)
            
        Y_test = Y_test_original   
        test_pred = test_pred_original
    
    # compute metrics
    test_mse = mean_squared_error(Y_test, test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(Y_test, test_pred)
    test_nmae = test_mae / (Y_max - Y_min)    
    r_squared = r2_score(Y_test, test_pred)
    
    print("Test MSE for Random Forest:", test_mse)
    print("Test RMSE for Random Forest:", test_rmse)
    print("Test MAE for Random Forest:", test_mae)
    print("Test NMAE for Random Forest:", test_nmae)
    #print("R-squared for Random Forest:", r_squared)
  
    result_pred = pd.DataFrame({
        "Observed_target": Y_test,  # Observed values
        "Predicted_target": test_pred  # Predicted values
    })

    #return mlp, test_mse, test_rmse, test_mae, test_nmae, r_squared, result_pred
    return mlp, result_pred    
    
                
def create_hexbin_plot(observed_values, predicted_values, y_range=None, gridsize = 60, cmap = 'pink_r', figsize = (8,6), title=''):
    plt.figure(figsize=figsize)
    
    hb = plt.hexbin(observed_values, predicted_values, gridsize=gridsize, cmap=cmap, mincnt=0)
    
    if y_range is not None:
        plt.xlim(y_range)
        plt.ylim(y_range)
    
    plt.xlabel('Observed Values')
    plt.ylabel('Predicted Values')
    plt.title(title, fontsize=18)
    
    plt.colorbar(label='Counts')
    
    plt.show()        
        
        
def compute_permutation_importances(rf_model, X_test, Y_test, n_repeats=1, random_state=42):   
	
    #check how long it takes for 1 repeat before increasing it

    # we compute permutation feature importances
    perm_importances = permutation_importance(rf_model, X_test, Y_test, n_repeats=n_repeats, random_state=random_state)
    importances = perm_importances.importances_mean
    feature_names = X_test.columns  
    
    # we store that
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False)

    return importances_df    
        
        
def standardize_importances(df_importances, multiply_by_100 = True):
    
    df_result = df_importances.copy()
    df_result.loc[df_result['Importance'] < 0, 'Importance'] = 0
    total_positive_importance = df_result['Importance'].sum()
    df_result['Importance'] = df_result['Importance'] / total_positive_importance
    if multiply_by_100:
        df_result['Importance'] = df_result['Importance'] * 100
        df_result['Importance'] = df_result['Importance'].round(1)

    return df_result
    
    
def analyze_importances_by_prefix(df_importances, prefixes,
                                  prefix_labels=None, title="Default Title",
                                  color1 = "blue", figsize =(10,6)):
    positive_sums = []
    total_sums = []
    
    if prefix_labels is None:
        prefix_labels = prefixes

    for prefix, label in zip(prefixes, prefix_labels):
        filtered_df = df_importances[df_importances['Feature'].str.startswith(prefix)]
        positive_sum = filtered_df[filtered_df['Importance'] > 0]['Importance'].sum()
        total_sum = filtered_df['Importance'].sum()

        positive_sums.append(positive_sum)
        total_sums.append(total_sum)

    plt.figure(figsize=figsize)
    ax1 = plt.subplot(1, 2, 1)
    ax1.barh(prefix_labels, positive_sums, color=color1, alpha=0.7)
    ax1.set_xlim([0, max(positive_sums) * 1.1])
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,.1f}".format(x)))
    ax1.set_xlabel('Importances (PFI sum =100)')
    ax1.set_title(title, fontsize = 18)

    ax1.tick_params(axis='y', labelsize=14)

    for i, v in enumerate(positive_sums):
        ax1.text(v, i, f'{v:.1f}', va='center', color='black', fontweight='normal')

    plt.show()        
        
        
def plot_top_features_scatter(df, marker = 'o', s = 100, label_f = '', title ='',  
                              fontsize_x =12, fontsize_y = 10, fontsize_t =18, 
                              x_label = 'PFI en %', 
                              y_label = '', 
                              labelsize = 15, 
                              x_min=None, x_max=None,
                              save_path = None, save_filename = None):
    
    top_features = df.sort_values(by='Importance', ascending=False).head(5)
    plt.figure(figsize=(4, 4))
    plt.scatter(top_features['Importance'], top_features['Feature'], color='dimgrey', marker=marker, s=s, label=label_f)

    plt.xlabel(x_label, fontsize=fontsize_x)
    plt.ylabel(y_label, fontsize=fontsize_y)
    plt.title(title, fontsize=fontsize_t)

    plt.gca().invert_yaxis()
    
    plt.tick_params(axis='y', labelsize=labelsize)

    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)

    plt.show()        
        
        
def generate_importances_heatmap(rf_importances, xgb_importances, nnet_importances, nb_values, title, annot=False, show_colorbar=False):

    # Step 1: we add rank columns to each DataFrame
    rf_importances['rf_rank'] = range(1, len(rf_importances) + 1)
    xgb_importances['xgb_rank'] = range(1, len(xgb_importances) + 1)
    nnet_importances['nnet_rank'] = range(1, len(nnet_importances) + 1)

    # Step 2: we rename 'Importances' columns
    rf_importances = rf_importances.rename(columns={'Importance': 'RF_importances'})
    xgb_importances = xgb_importances.rename(columns={'Importance': 'XGB_importances'})
    nnet_importances = nnet_importances.rename(columns={'Importance': 'MLP_importances'})

    # Step 3: we merge the three DataFrames on the 'Feature' column
    model_importances = pd.merge(rf_importances, xgb_importances, on='Feature', how='inner')
    model_importances = pd.merge(model_importances, nnet_importances, on='Feature', how='inner')

    # we calculate the minimum rank among rf_rank, xgb_rank, and nnet_rank
    model_importances['min_rank'] = model_importances[['rf_rank', 'xgb_rank', 'nnet_rank']].min(axis=1)

    # we sort the DataFrame by 'min_rank' and keep the top 'nb_values' rows
    top_importances = model_importances.sort_values(by='min_rank').head(nb_values)
    
    
    top_importances['RF_importances'] = top_importances['RF_importances'].clip(lower=0)
    top_importances['XGB_importances'] = top_importances['XGB_importances'].clip(lower=0)
    top_importances['MLP_importances'] = top_importances['MLP_importances'].clip(lower=0)

    top_importances_subset = top_importances[['Feature', 'RF_importances', 'XGB_importances', 'MLP_importances']]

    heatmap_data = top_importances_subset.set_index('Feature')
    heatmap_data_log = np.log1p(heatmap_data*10)  
    plt.figure(figsize=(3, 15))
    ax = sns.heatmap(heatmap_data_log, cmap='viridis', annot=annot, fmt=".1f", linewidths=0.5)
    plt.title(title, fontsize = 18)
    
    if not show_colorbar:
        ax.collections[0].colorbar.remove()
        
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha='right')

    plt.show()        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
