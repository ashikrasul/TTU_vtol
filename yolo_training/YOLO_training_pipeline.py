import os
import yaml
import numpy as np
from ultralytics import YOLO, settings as ultralytics_settings
import shutil
from utils import constants  # Ensure constants.metadata_file_path is defined

# Ultralytics gates its Comet integration on the persistent setting
# SETTINGS["comet"] (~/.config/Ultralytics/settings.yaml), checked at
# training start — not on the COMET_MODE env var. COMET_MODE only chooses
# online vs. offline *after* the integration is already active, and
# 'disabled' isn't a recognized value (falls through to a live online
# Experiment). Disabling it here is what actually prevents comet_ml from
# spinning up an experiment and hanging on asset-upload "delivery
# confirmation" for minutes on every training run.
if ultralytics_settings.get('comet') is not False:
    ultralytics_settings.update({'comet': False})

def create_incremental_folder(base_dir, prefix='Train'):
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while True:
        new_folder = os.path.join(base_dir, f"{prefix}{i}")
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            return new_folder  # Exit after creating folder
        i += 1

class YOLOTrainingPipeline:
    def __init__(self, cfg_file, save_dir, scale_value, hsv_v_value, base_weights='yolov8s.pt'):
        self.cfg_file = cfg_file
        self.save_dir = save_dir
        self.scale_value = self.convert_numpy(scale_value)  # Set scale value
        self.hsv_v_value = self.convert_numpy(hsv_v_value)  # Set hsv_v value
        self.base_weights = base_weights  # Starting weights for model.train()

        # Required by torch.use_deterministic_algorithms() for deterministic
        # cuBLAS matmul/conv behavior — must be set before any CUDA context
        # is created. The actual seed (0) and deterministic flag come from
        # cfg_file (hyp_bayes.yaml) and are applied by ultralytics' own
        # init_seeds() inside model.train(); setting a seed here would only
        # be immediately overwritten, so we don't duplicate it.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    def convert_numpy(self, obj):
        """Convert NumPy objects to standard Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    def get_best_model_path(self):
        """Generates the next available model name in the save directory."""
        os.makedirs(self.save_dir, exist_ok=True)
        existing_models = [f for f in os.listdir(self.save_dir) if f.startswith("yolo") and f.endswith(".pt")]
        model_numbers = [int(f[4:-3]) for f in existing_models if f[4:-3].isdigit()]
        next_number = max(model_numbers, default=0) + 1
        return os.path.join(self.save_dir, f"yolo{next_number}.pt")

    def train_yolov8(self):
        """Trains YOLOv8 with the specified scale and hsv_v, and saves only the best model."""
        results_dir = './training_result'
        save_dir = create_incremental_folder(results_dir)

        # Load and update the configuration file
        with open(self.cfg_file, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        config.update({
            'project': save_dir,
            'name': 'train',
            'exist_ok': True,
            'scale': self.scale_value,
            'hsv_v': self.hsv_v_value
        })

        # Save the modified configuration to a temporary file
        temp_cfg_file = os.path.join(save_dir, 'temp_config.yaml')
        with open(temp_cfg_file, 'w') as file:
            yaml.dump(config, file)

        # Train YOLOv8 model
        model = YOLO(self.base_weights)  # Load the YOLOv8 model with starting weights
        model.train(cfg=temp_cfg_file)

        # Save only the best model
        best_model_path = os.path.join(save_dir, 'train', 'weights', 'best.pt')
        if os.path.exists(best_model_path):
            final_model_path = self.get_best_model_path()
            os.rename(best_model_path, final_model_path)
            print(f"Best model saved as: {final_model_path}")

            # Save metadata
            self.save_metadata(
                scale=self.scale_value,
                hsv_v=self.hsv_v_value,
                model_name=os.path.basename(final_model_path),
                training_folder=os.path.basename(save_dir),
            )
        else:
            print(f"Best model not found at: {best_model_path}")

        # Clean up intermediate files
        #self.cleanup(save_dir=os.path.join(save_dir, 'train'))

    def save_metadata(self, scale, hsv_v, model_name, training_folder=None):
        """Append metadata to a YAML file."""
        os.makedirs(os.path.dirname(constants.metadata_file_path), exist_ok=True)

        metadata = {
            'scale': self.convert_numpy(scale),
            'hsv_v': self.convert_numpy(hsv_v),
            'model_name': str(model_name),
        }
        if training_folder is not None:
            metadata['training_folder'] = str(training_folder)

        if os.path.exists(constants.metadata_file_path):
            with open(constants.metadata_file_path, 'r') as file:
                try:
                    existing_data = yaml.safe_load(file) or {}
                except yaml.YAMLError:
                    existing_data = {}
        else:
            existing_data = {}

        existing_data.update(metadata)

        with open(constants.metadata_file_path, 'w') as file:
            yaml.dump(existing_data, file)

        print(f"Training metadata appended to {constants.metadata_file_path}")

    def cleanup(self, save_dir):
        """Remove all files except the best model."""
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir, ignore_errors=True)

    def run(self):
        self.train_yolov8()
