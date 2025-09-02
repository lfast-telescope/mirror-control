# LFAST Mirror Control System

Integrated active optics control system for the LFAST telescope, combining multiple wavefront sensing methods and mirror control capabilities.

## Overview

This repository integrates three main wavefront sensing approaches:

- **Interferometer**: Phase-shifting interferometry for high-precision surface measurements
- **SHWFS**: Shack-Hartmann wavefront sensor for real-time sensing
- **Phase Retrieval**: Phase diversity method for wavefront reconstruction

## Repository Structure

```
mirror_control/
├── interferometer/          # Interferometer submodule
├── SHWFS/                  # SHWFS submodule  
├── phase_retrieval/        # Phase retrieval submodule
├── shared/                 # Shared utilities and interfaces
│   ├── base_classes.py     # Abstract base classes
│   ├── config_manager.py   # Configuration management
│   ├── hardware/           # Hardware abstraction layer
│   ├── analysis/           # Shared analysis tools
│   └── plotting/           # Shared plotting utilities
├── config/                 # Configuration files
├── main.py                 # Main integration script
└── requirements.txt        # Python dependencies
```

## Setup

1. **Clone with submodules:**
   ```bash
   git clone --recursive https://github.com/lfast-telescope/mirror-control.git
   cd mirror-control
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize submodules (if not cloned with --recursive):**
   ```bash
   git submodule update --init --recursive
   ```

4. **Create default configurations:**
   ```python
   from shared.config_manager import ConfigManager
   config_mgr = ConfigManager()
   config_mgr.create_default_configs()
   ```

## Usage

### Basic Integration Test
```bash
python main.py
```

### Using Individual Measurement Methods

#### Interferometer
```python
from interferometer.main import main as interferometer_main
# Configure and run interferometer measurement
```

#### SHWFS
```python
import SHWFS
# Use SHWFS for real-time measurements
```

#### Phase Retrieval
```python
import phase_retrieval
# Use phase diversity method
```

### Active Optics Control Loop
```python
from shared.base_classes import ActiveOpticsLoop
from your_sensor_implementation import YourSensor
from your_controller_implementation import YourController

sensor = YourSensor()
controller = YourController()
loop = ActiveOpticsLoop(sensor, controller)

# Run single iteration
result = loop.run_single_iteration()

# Run closed loop
loop.run_closed_loop(max_iterations=10)
```

## Configuration

Configuration files are stored in the `config/` directory in YAML format. Each measurement method has its own configuration file:

- `interferometer.yaml`: Interferometer settings
- `shwfs.yaml`: SHWFS settings  
- `phase_retrieval.yaml`: Phase retrieval settings

## Development

Each submodule is a separate repository that can be developed independently:

- [interferometer](https://github.com/lfast-telescope/interferometer)
- [SHWFS](https://github.com/lfast-telescope/SHWFS)
- [phase_retrieval](https://github.com/lfast-telescope/phase_retrieval)

To update submodules:
```bash
git submodule update --remote
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes in the appropriate submodule or shared code
4. Test integration
5. Submit a pull request

## License

See individual submodule repositories for their respective licenses.
