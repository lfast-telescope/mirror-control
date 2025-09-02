"""
Abstract base classes for wavefront measurement and control.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional

class WavefrontSensor(ABC):
    """Abstract base class for all wavefront sensing methods."""
    
    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the sensor with given parameters."""
        pass
    
    @abstractmethod
    def acquire(self) -> np.ndarray:
        """Acquire raw measurement data."""
        pass
    
    @abstractmethod
    def process(self, raw_data: np.ndarray) -> np.ndarray:
        """Process raw data into wavefront map."""
        pass
    
    @abstractmethod
    def get_zernike_coefficients(self, wavefront: np.ndarray) -> np.ndarray:
        """Extract Zernike coefficients from wavefront."""
        pass

class MirrorController(ABC):
    """Abstract base class for mirror control systems."""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to mirror hardware."""
        pass
    
    @abstractmethod
    def apply_correction(self, zernike_coeffs: np.ndarray) -> bool:
        """Apply correction based on Zernike coefficients."""
        pass
    
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get current mirror status."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from mirror hardware."""
        pass

class ActiveOpticsLoop:
    """Main active optics control loop integrating sensors and actuators."""
    
    def __init__(self, sensor: WavefrontSensor, controller: MirrorController):
        self.sensor = sensor
        self.controller = controller
        self.is_running = False
    
    def run_single_iteration(self) -> Dict[str, Any]:
        """Run one iteration of the active optics loop."""
        # Acquire measurement
        raw_data = self.sensor.acquire()
        
        # Process to get wavefront
        wavefront = self.sensor.process(raw_data)
        
        # Extract Zernike coefficients
        zernike_coeffs = self.sensor.get_zernike_coefficients(wavefront)
        
        # Apply correction
        success = self.controller.apply_correction(zernike_coeffs)
        
        return {
            'wavefront': wavefront,
            'zernike_coefficients': zernike_coeffs,
            'correction_applied': success,
            'controller_status': self.controller.get_status()
        }
    
    def run_closed_loop(self, max_iterations: int = 10, 
                       convergence_threshold: float = 0.1) -> None:
        """Run closed-loop active optics correction."""
        self.is_running = True
        
        for i in range(max_iterations):
            if not self.is_running:
                break
                
            result = self.run_single_iteration()
            
            # Check for convergence
            rms_error = np.sqrt(np.mean(result['zernike_coefficients']**2))
            print(f"Iteration {i+1}: RMS error = {rms_error:.3f}")
            
            if rms_error < convergence_threshold:
                print(f"Converged after {i+1} iterations")
                break
    
    def stop(self) -> None:
        """Stop the active optics loop."""
        self.is_running = False
