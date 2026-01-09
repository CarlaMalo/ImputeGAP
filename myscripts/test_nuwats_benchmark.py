import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from imputegap.recovery.benchmark import Benchmark

my_algorithms = ["NuwaTS2","NuwaTS","GPT4TS","BRITS","MeanImpute"]

my_opt = ["default_params"]

#my_datasets = ["eeg-alcohol","stock-exchange","airq"]
my_datasets = ["airq"]
my_patterns = ["scattered"]  # "scattered" or "mcar"

cont_per_series = 0.3
if my_patterns[0] == "scattered":
    range = [[cont_per_series,0.01],[cont_per_series,0.05],[cont_per_series,0.10],[cont_per_series,0.15]]
elif my_patterns[0] == "mcar":
    range = [[cont_per_series,0.2],[cont_per_series,0.4],[cont_per_series,0.6],[cont_per_series,0.8]]
 
my_metrics = ["*"]

# launch the evaluation
bench = Benchmark()
bench.eval(algorithms=my_algorithms, datasets=my_datasets, patterns=my_patterns, x_axis=range, metrics=my_metrics, optimizers=my_opt, save_dir="./imputegap_NEW_assets/benchmark_no_orig_mask", verbose=True)