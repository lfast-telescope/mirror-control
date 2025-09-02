"""
Configuration management for the LFAST mirror control system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages configuration for all measurement methods and hardware."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            # Try JSON
            config_file = self.config_dir / f"{config_name}.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                if config_file.suffix == '.yaml':
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
            
            self._configs[config_name] = config
            return config
        else:
            raise FileNotFoundError(f"Config file not found: {config_name}")
    
    def save_config(self, config_name: str, config_data: Dict[str, Any]) -> None:
        """Save configuration to file."""
        config_file = self.config_dir / f"{config_name}.yaml"
        
        with open(config_file, 'w') as f:
            yaml.safe_dump(config_data, f, default_flow_style=False)
        
        self._configs[config_name] = config_data
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """Get configuration (load if not already loaded)."""
        if config_name not in self._configs:
            return self.load_config(config_name)
        return self._configs[config_name]
    
    def create_default_configs(self) -> None:
        """Create default configuration files."""
        
        # Interferometer config
        interferometer_config = {
            'hardware': {
                'smc100_port': 'COM3',
                'smc100_channels': 3,
                'camera': {
                    'exposure_time': 100,
                    'gain': 1.0
                }
            },
            'processing': {
                'crop_size': [512, 512],
                'phase_steps': 4,
                'unwrap_method': 'goldstein'
            },
            'zernike': {
                'max_order': 10,
                'aperture_diameter': 100
            }
        }
        
        # SHWFS config
        shwfs_config = {
            'hardware': {
                'camera_id': 0,
                'lenslet_pitch': 150e-6,
                'focal_length': 5e-3
            },
            'processing': {
                'spot_detection_threshold': 0.1,
                'centroid_method': 'center_of_mass',
                'reference_centroids_file': 'reference_centroids.npy'
            },
            'zernike': {
                'max_order': 15,
                'aperture_diameter': 100
            }
        }
        
        # Phase retrieval config
        phase_retrieval_config = {
            'hardware': {
                'stage_controller': 'newport_smc100',
                'defocus_positions': [-2e-3, 0, 2e-3],
                'camera_settings': {
                    'exposure_time': 50,
                    'gain': 2.0
                }
            },
            'processing': {
                'pupil_diameter': 256,
                'iterations': 50,
                'convergence_threshold': 1e-6
            },
            'zernike': {
                'max_order': 12,
                'aperture_diameter': 100
            }
        }
        
        # Save all configs
        self.save_config('interferometer', interferometer_config)
        self.save_config('shwfs', shwfs_config)
        self.save_config('phase_retrieval', phase_retrieval_config)
        
        print("Default configuration files created in:", self.config_dir)
