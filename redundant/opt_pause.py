import os
import gc
import pandas as pd
import subprocess
import torch
from bayes_opt import BayesianOptimization
from yolo_training.YOLO_training_pipeline import YOLOTrainingPipeline
from utils import constants


def get_next_optimization_run_number(meta_file_path):
    os.makedirs(os.path.dirname(meta_file_path), exist_ok=True)

    last_run_number = 0
    if os.path.exists(meta_file_path):
        with open(meta_file_path, 'r') as f:
            for line in f:
                if line.startswith("optimization_run:"):
                    last_run_number = int(line.split(":")[1].strip())
                    break

    next_run_number = last_run_number + 1

    with open(meta_file_path, 'w') as f:
        f.write(f"optimization_run: {next_run_number}\n")

    print(f"Updated optimization run number to: {next_run_number}")
    return next_run_number

def get_latest_success_rate(results_csv):
    try:
        df = pd.read_csv(results_csv)
        df['Run Number'] = df['Run Number'].str.extract('(\d+)').astype(int)
        latest_run = df[df['Run Number'] == df['Run Number'].max()].iloc[0]
        success_rate = latest_run['Success Rate']
        print(f"Success rate for latest run (Run {latest_run['Run Number']}): {success_rate}")
        return success_rate
    except Exception as e:
        print(f"Error reading success rate: {e}")
        return 0.0  # Default to 0 if error occurs

def train_and_evaluate(scale, hsv_v):
    cfg_file = './configs/hyp_bayes.yaml'
    results_csv = './utils/performance_summary.csv'
    save_dir = './perception/yolov5/models'

    try:
        torch.cuda.empty_cache()

        pipeline = YOLOTrainingPipeline(cfg_file=cfg_file, save_dir=save_dir, scale_value=scale, hsv_v_value=hsv_v)
        pipeline.run()

        torch.cuda.empty_cache()
        gc.collect()

        subprocess.run(["python3", "rraaa.py", "configs/single-static.yml"], check=True)

        torch.cuda.empty_cache()
        gc.collect()

        return get_latest_success_rate(results_csv)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Cleaning up resources...")
        torch.cuda.empty_cache()
        gc.collect()
        return 0.0


def load_posterior_points_for_run(file_path, run_number, max_points=30):
    """
    Load posterior values (Scale, HSV_V, and Success Rate) for a specific optimization run from the CSV file.
    """
    try:
        df = pd.read_csv(file_path)

        # Filter rows corresponding to the specified optimization run
        filtered_points = df[df['Optimization Run'] == run_number]

        # Select the required columns and limit to the first `max_points` rows
        selected_points = filtered_points[['Scale', 'HSV_V', 'Success Rate']].head(max_points).to_dict(orient='records')

        print(f"Loaded posterior points for Optimization Run {run_number}: {selected_points}")
        return selected_points
    except Exception as e:
        print(f"Error loading posterior points for Optimization Run {run_number}: {e}")
        return []


def optimize_scale_and_hsv_v(posterior_points):
    pbounds = {
        'scale': (0.0, 1),
        'hsv_v': (0.0, 1)
    }

    optimizer = BayesianOptimization(
        f=train_and_evaluate,
        pbounds=pbounds,
        verbose=2,
        random_state=42
    )

    # Register posterior points into the optimizer
    for point in posterior_points:
        optimizer.register(
            params={'scale': point['Scale'], 'hsv_v': point['HSV_V']},
            target=point['Success Rate']
        )

    try:
        optimizer.maximize(
            init_points=5 - len(posterior_points),  # Adjust based on registered points
            n_iter=30
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Exiting gracefully...")
        return optimizer.max

    print(f"Best parameters found: {optimizer.max}")
    return optimizer.max


if __name__ == "__main__":
    try:
        get_next_optimization_run_number(constants.metadata_file_path)

        posterior_points_csv = './utils/performance_summary.csv'
        optimization_run = 15  # Specify the optimization run
        posterior_points = load_posterior_points_for_run(posterior_points_csv, optimization_run)

        best_params = optimize_scale_and_hsv_v(posterior_points)

        print(f"Optimized parameters: scale={best_params['params']['scale']}, hsv_v={best_params['params']['hsv_v']}")
    except KeyboardInterrupt:
        print("\nScript terminated by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
