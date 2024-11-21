import os
import gc
import pandas as pd
import subprocess
import torch
from bayes_opt import BayesianOptimization
from yolo_training.YOLO_training_pipeline import YOLOTrainingPipeline
from utils import constants



def get_next_optimization_run_number(meta_file_path):
    """
    Reads the last optimization run number from the metadata file, increments it, and saves it back.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(meta_file_path), exist_ok=True)

    # Read the last run number from the metadata file
    last_run_number = 0
    if os.path.exists(meta_file_path):
        with open(meta_file_path, 'r') as f:
            for line in f:
                if line.startswith("optimization_run:"):
                    last_run_number = int(line.split(":")[1].strip())
                    break

    # Increment the run number
    next_run_number = last_run_number + 1

    # Update the metadata file with the new run number
    with open(meta_file_path, 'w') as f:
        f.write(f"optimization_run: {next_run_number}\n")

    print(f"Updated optimization run number to: {next_run_number}")
    return next_run_number


def get_latest_success_rate(results_csv):
    df = pd.read_csv(results_csv)
    df['Run Number'] = df['Run Number'].str.extract('(\d+)').astype(int)
    latest_run = df[df['Run Number'] == df['Run Number'].max()].iloc[0]
    success_rate = latest_run['Success Rate']
    print(f"Success rate for latest run (Run {latest_run['Run Number']}): {success_rate}")
    return success_rate

def train_and_evaluate(scale, hsv_v):
    cfg_file = '/home/arasul42@tntech.edu/works/rraaa-sim/configs/hyp_bayes.yaml'
    results_csv = '/home/arasul42@tntech.edu/works/rraaa-sim/utils/performance_summary.csv'

    # Directory to save trained YOLO models
    save_dir = '/home/arasul42@tntech.edu/works/temp/rraaa-sim/perception/yolov5/models'

    torch.cuda.empty_cache()

    # Pass the hyperparameters (scale and hsv_v) to the YOLOTrainingPipeline
    pipeline = YOLOTrainingPipeline(cfg_file=cfg_file, save_dir=save_dir, scale_value=scale, hsv_v_value=hsv_v)
    pipeline.run()

    # Run the external script to calculate landing success rate
    subprocess.run(["python3", "rraaa.py", "configs/single-static.yml"], check=True)

    torch.cuda.empty_cache()
    gc.collect()

    return get_latest_success_rate(results_csv)

def optimize_scale_and_hsv_v():
    """
    Perform Bayesian Optimization on the `scale` and `hsv_v` hyperparameters using `train_and_evaluate` function.
    """
    # Define the bounds for the `scale` and `hsv_v` hyperparameters
    pbounds = {
        'scale': (0.0, 1),  # Adjust bounds based on expected scale range
        'hsv_v': (0.0, 1)   # Example bounds for hsv_v
    }

    # Instantiate the optimizer
    optimizer = BayesianOptimization(
        f=train_and_evaluate,  # The function to optimize
        pbounds=pbounds,       # Parameter bounds
        verbose=2,             # Display progress
        random_state=42        # Ensure reproducibility
    )

    try:
        # Run the optimization
        optimizer.maximize(
            init_points=3,  # Number of random initial points
            n_iter=15       # Number of optimization iterations
        )

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Exiting gracefully...")
        return optimizer.max

    # Output the best result
    print(f"Best parameters found: {optimizer.max}")
    return optimizer.max

if __name__ == "__main__":
    try:

        
        get_next_optimization_run_number(constants.metadata_file_path)
        
        best_params = optimize_scale_and_hsv_v()
        print(f"Optimized parameters: scale={best_params['params']['scale']}, hsv_v={best_params['params']['hsv_v']}")
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
        