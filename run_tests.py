#!/usr/bin/env python3
"""
Comprehensive test runner for mBART seq-to-seq project
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Failed with return code: {result.returncode}")
        return False
    else:
        print("✅ Passed")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run mBART project tests")
    parser.add_argument("--type", choices=["unit", "integration", "performance", "regression", "all"], 
                       default="all", help="Type of tests to run")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--gpu", action="store_true", help="Run GPU-specific tests")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print(f"🚀 Running mBART Project Tests")
    print(f"Working directory: {project_dir}")
    print(f"Test type: {args.type}")
    print(f"Coverage: {args.coverage}")
    print(f"Fast mode: {args.fast}")
    
    # Base pytest command
    cmd_parts = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd_parts.append("-v")
    else:
        cmd_parts.append("-q")
    
    # Add coverage
    if args.coverage:
        cmd_parts.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    # Add parallel execution
    if args.parallel:
        cmd_parts.extend(["-n", str(args.parallel)])
    
    # Test markers and paths based on type
    if args.type == "unit":
        cmd_parts.extend(["tests/unit/", "-m", "unit"])
    elif args.type == "integration":
        cmd_parts.extend(["tests/integration/", "-m", "integration"])
    elif args.type == "performance":
        cmd_parts.extend(["tests/performance/", "-m", "performance"])
    elif args.type == "regression":
        cmd_parts.extend(["tests/regression/", "-m", "regression"])
    elif args.type == "all":
        cmd_parts.append("tests/")
    
    # Skip slow tests if requested
    if args.fast:
        cmd_parts.extend(["-m", "not slow"])
    
    # Add GPU tests if requested
    if args.gpu:
        cmd_parts.extend(["-m", "gpu or not gpu"])
    else:
        cmd_parts.extend(["-m", "not gpu"])
    
    # Join command
    cmd = " ".join(cmd_parts)
    
    # Run tests
    success = run_command(cmd, f"{args.type.title()} Tests")
    
    if success:
        print(f"\n🎉 All {args.type} tests passed!")
        
        if args.coverage:
            print(f"\n📊 Coverage report generated in htmlcov/")
            
    else:
        print(f"\n❌ Some {args.type} tests failed!")
        sys.exit(1)
    
    # Additional commands for comprehensive testing
    if args.type == "all":
        print(f"\n📋 Running additional checks...")
        
        # Linting (if available)
        if os.path.exists("requirements.txt"):
            try:
                subprocess.run(["python", "-m", "flake8", "src/", "--max-line-length=100"], 
                             check=False, capture_output=True)
                print("✅ Code style check completed")
            except FileNotFoundError:
                print("⚠️  flake8 not available for style checking")
        
        # Type checking (if available)
        try:
            subprocess.run(["python", "-m", "mypy", "src/", "--ignore-missing-imports"], 
                         check=False, capture_output=True)
            print("✅ Type checking completed")
        except FileNotFoundError:
            print("⚠️  mypy not available for type checking")
    
    print(f"\n{'='*60}")
    print(f"Test run completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()