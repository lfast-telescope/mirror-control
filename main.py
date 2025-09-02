#!/usr/bin/env python3
"""
LFAST Mirror Control System
Main integration script for active optics control using multiple wavefront sensing methods.
"""

import sys
import os
from pathlib import Path

# Add submodules to path
PROJECT_ROOT = Path(__file__).parent
sys.path.extend([
    str(PROJECT_ROOT / "interferometer"),
    str(PROJECT_ROOT / "SHWFS"), 
    str(PROJECT_ROOT / "phase_retrieval"),
    str(PROJECT_ROOT / "shared")
])

def main():
    """Main entry point for mirror control system."""
    print("LFAST Mirror Control System")
    print("Available measurement methods:")
    print("1. Interferometer")
    print("2. SHWFS")
    print("3. Phase Retrieval")
    
    # Import and test submodules
    try:
        from interferometer.main import main as interferometer_main
        print("✓ Interferometer module loaded")
    except ImportError as e:
        print(f"✗ Interferometer module error: {e}")
    
    try:
        # Test SHWFS import (adjust based on actual module structure)
        import SHWFS
        print("✓ SHWFS module loaded")
    except ImportError as e:
        print(f"✗ SHWFS module error: {e}")
    
    try:
        # Test phase retrieval import (adjust based on actual module structure)
        import phase_retrieval
        print("✓ Phase retrieval module loaded")
    except ImportError as e:
        print(f"✗ Phase retrieval module error: {e}")

if __name__ == "__main__":
    main()
