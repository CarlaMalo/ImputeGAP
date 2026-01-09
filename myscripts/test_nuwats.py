from os import name
import sys
from pathlib import Path
import os
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from imputegap.recovery.imputation import Imputation
from imputegap.recovery.manager import TimeSeries
from imputegap.tools import utils

cont_type = "MCAR" # "scattered" or "MCAR"
print(f"Contamination type: {cont_type}")
des = "trim" # trimmed to nbr_vals=2500
main_path = f"./imputegap_NEW_assets/imputation_{des}_multirun_" + cont_type
# load dataset
dataset_list = ["airq","eeg-alcohol", "chlorine", "stock-exchange"]  #"custom/21_weather.txt"
results = {}

for dataset_name in dataset_list:
    print(f"\nProcessing dataset: {dataset_name}\n")
    # initialize the time series object
    ts = TimeSeries()
    ts.load_series(utils.search_path(dataset_name), nbr_val=2500) 
    ts.normalize(normalizer="z_score")
    
    cont_dataset = 0.3 # Fixed this to 30% missing per series

    if cont_type == "scattered":
        if dataset_name == "eeg-alcohol":
            cont = [0.05]
        if dataset_name == "airq":
            cont = [0.15]
        if dataset_name == "chlorine":
            cont = [0.25]
        if dataset_name == "stock-exchange":
            cont = [0.08]
    else: # MCAR
        cont = [0.2,0.5]
        
    for rate_series in cont:
        # Multiple runs with different contamination
        metrics_per_run = []  # Store metrics from all runs
        
        for run in range(3):
            print(f" Run {run+1} / 3")
            np.random.seed(42 + run)  # Set seed for reproducibility
            print(f"  Contamination rate per series: {rate_series}")  
            
            # contaminate the time series
            if cont_type == "scattered":
                ts_m = ts.Contamination.scattered(ts.data, rate_dataset=cont_dataset, rate_series=rate_series, seed=False)
            else:
                ts_m = ts.Contamination.mcar(ts.data, rate_dataset=cont_dataset, rate_series=rate_series, block_size=1, seed=False)
            # impute the contaminated series
            imputer = Imputation.LLMs.NuwaTS2(ts_m)
            tr_ratio = (1 - cont_dataset*rate_series - 0.06) # Hardcoded to not add more contamination to test mask
            #imputer = imputer.impute(user_def=True, params={"seq_length":-1, "patch_size":-1, "batch_size":-1, "gpt_layers":6, "num_workers":0, "seed":42},tr_ratio=tr_ratio, original_tr_mask=True, data_name=dataset_name)
            imputer = imputer.impute(user_def=True, params={"seq_length":96, "patch_size":16, "batch_size":-1, "gpt_layers":6, "num_workers":0, "seed":42},tr_ratio=tr_ratio, original_tr_mask=True, data_name=dataset_name)

            # Compute and print the imputation metrics
            imputer.score(ts.data, imputer.recov_data)
            ts.print_results(imputer.metrics)
            metrics_per_run.append(imputer.metrics.copy())  # Store metrics from this run

            # Display a portion of the time series
            start_interval = int(ts.data.shape[1]*0.1)
            end_interval = start_interval + 300
            interval = (start_interval, end_interval)
            ts.plot(input_data=ts.data[:,interval[0]:interval[1]], incomp_data=ts_m[:,interval[0]:interval[1]], recov_data=imputer.recov_data[:,interval[0]:interval[1]], nbr_series=2, cont_rate=str(rate_series*100), subplot=True, algorithm=imputer.algorithm, save_path=f"{main_path}/{dataset_name}/imputation_run_{run}")
            # Save individual run results
            os.makedirs(f"{main_path}/{dataset_name}", exist_ok=True)
            with open(f"{main_path}/{dataset_name}/results.txt",
                        "a") as result_file:
                    result_file.write(f"Dataset: {dataset_name}, Contamination rate per series: {rate_series}, Run: {run+1}\n")
                    for metric_name, metric_value in imputer.metrics.items():
                        result_file.write(f"{metric_name}: {metric_value}\n")
                    result_file.write("\n")
        
        # Calculate and write aggregate results (mean and std across runs)
        os.makedirs(f"{main_path}/{dataset_name}", exist_ok=True)
        with open(f"{main_path}/{dataset_name}/results.txt",
                    "a") as result_file:
            result_file.write(f"\n{'='*60}\n")
            result_file.write(f"AGGREGATE RESULTS - Dataset: {dataset_name}, Contamination rate: {rate_series}\n")
            result_file.write(f"{'='*60}\n")
            
            # Get all metric names from the first run
            metric_names = list(metrics_per_run[0].keys())
            
            # Calculate mean and std for each metric
            for metric_name in metric_names:
                values = [metrics[metric_name] for metrics in metrics_per_run]
                mean_val = np.mean(values)
                std_val = np.std(values)
                result_file.write(f"{metric_name:20s} = Mean: {mean_val:.16f}, Std: {std_val:.16f}\n")
            
            result_file.write(f"{'='*60}\n\n")
