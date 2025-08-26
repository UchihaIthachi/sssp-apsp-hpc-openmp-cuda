#!/usr/bin/env python3
import subprocess
import re
import json
import argparse
import os

def run_command(command):
    """Runs a command and returns its stdout."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Stderr: {e.stderr}")
        return None

def parse_time(output, pattern):
    """Parses the execution time from the output using a regex."""
    match = re.search(pattern, output)
    if match:
        return float(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Run Bellman-Ford benchmarks.")
    parser.add_argument(
        '--vertices',
        nargs='+',
        type=int,
        default=[1000, 2000, 5000, 10000],
        help="A list of vertex counts to benchmark."
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='benchmark_results.json',
        help="The file to save the benchmark results to."
    )
    args = parser.parse_args()

    # Check if nvcc is available
    nvcc_available = run_command("nvcc --version") is not None

    # Build the project first
    print("Building project...")
    build_cmd = "make all"
    if not nvcc_available:
        print("nvcc not found. Building only serial and OpenMP targets.")
        build_cmd = "make bin/BF_serial bin/BF_openmp"
    
    if run_command(build_cmd) is None:
        print("Make failed. Aborting.")
        return

    results = []

    for v in args.vertices:
        print(f"Running benchmarks for {v} vertices...")
        
        result_row = {'vertices': v}

        # --- Serial ---
        print("  Running Serial...")
        serial_cmd = f"./bin/BF_serial {v} -30 30 0.001"
        serial_output = run_command(serial_cmd)
        if serial_output:
            serial_time = parse_time(serial_output, r"\[serial\] time: ([\d.]+) s")
            result_row['serial_time'] = serial_time
            print(f"    Time: {serial_time}s")

        # --- OpenMP ---
        print("  Running OpenMP...")
        omp_cmd = f"OMP_NUM_THREADS=8 ./bin/BF_openmp {v} -30 30 0.001 8"
        omp_output = run_command(omp_cmd)
        if omp_output:
            omp_time = parse_time(omp_output, r"\[openmp\] time: ([\d.]+) s")
            result_row['openmp_time'] = omp_time
            print(f"    Time: {omp_time}s")

        # --- CUDA ---
        if nvcc_available:
            print("  Running CUDA...")
            cuda_cmd = f"./bin/BF_cuda {v} -30 30 0.001"
            cuda_output = run_command(cuda_cmd)
            if cuda_output:
                cuda_time = parse_time(cuda_output, r"\[cuda\] time: ([\d.]+) s")
                result_row['cuda_time'] = cuda_time
                print(f"    Time: {cuda_time}s")
        else:
            print("  Skipping CUDA (nvcc not found).")
            result_row['cuda_time'] = None

        # --- Hybrid ---
        if nvcc_available:
            print("  Running Hybrid...")
            hybrid_cmd = f"OMP_NUM_THREADS=8 ./bin/BF_hybrid {v} -30 30 0.5 0.001 8"
            hybrid_output = run_command(hybrid_cmd)
            if hybrid_output:
                hybrid_time = parse_time(hybrid_output, r"\[hybrid\] time: ([\d.]+) s")
                result_row['hybrid_time'] = hybrid_time
                print(f"    Time: {hybrid_time}s")
        else:
            print("  Skipping Hybrid (nvcc not found).")
            result_row['hybrid_time'] = None

        results.append(result_row)

    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print("Benchmark run complete.")

if __name__ == "__main__":
    main()
