#!/usr/bin/env python3
"""
Quick demonstration of the Autonomous System Validation capabilities.

This script provides a simple example of how to use the comprehensive
validation system for autonomous SeedAI operations.

Usage:
    python demo_validation.py [--scenario SCENARIO] [--duration MINUTES]

Examples:
    python demo_validation.py --scenario basic --duration 5
    python demo_validation.py --scenario evolution --duration 10
    python demo_validation.py --scenario performance --duration 8
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from validate_autonomous_system import AutonomousSystemValidator


def main():
    """Run a quick validation demonstration."""
    parser = argparse.ArgumentParser(description="Quick Validation Demo")
    parser.add_argument('--scenario', choices=['basic', 'evolution', 'performance', 'safety'],
                       default='basic', help='Validation scenario')
    parser.add_argument('--duration', type=int, default=5,
                       help='Duration in minutes')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')

    args = parser.parse_args()

    print(f"🚀 Starting {args.scenario} validation demo ({args.duration} minutes)")

    # Create validator with demo settings
    validator = AutonomousSystemValidator(
        config_path="configs/sample.yaml",
        duration_minutes=args.duration,
        monitoring_interval=3.0,  # Faster for demo
        scenario=args.scenario,
        enable_visualization=not args.no_visualize,
        verbose=True
    )

    try:
        # Run validation
        validator.start_validation()
        print("✅ Validation demo completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())