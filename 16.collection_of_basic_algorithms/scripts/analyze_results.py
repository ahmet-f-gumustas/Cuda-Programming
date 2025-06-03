#!/usr/bin/env python3
"""
Performance results analysis script for CUDA Parallel Algorithms Collection
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

def load_benchmark_results(filename):
    """Load benchmark results from CSV file"""
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def plot_scaling_analysis(df):
    """Create scaling analysis plots"""
    algorithms = df['Algorithm'].unique()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CUDA Algorithms Performance Analysis', fontsize=16)
    
    # Time vs Size
    ax1 = axes[0, 0]
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        ax1.loglog(algo_data['Size'], algo_data['Time_ms'], 'o-', label=algo, linewidth=2)
    ax1.set_xlabel('Problem Size (elements)')
    ax1.set_ylabel('Execution Time (ms)')
    ax1.set_title('Execution Time vs Problem Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Throughput vs Size
    ax2 = axes[0, 1]
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        ax2.semilogx(algo_data['Size'], algo_data['Throughput_GOPS'], 'o-', label=algo, linewidth=2)
    ax2.set_xlabel('Problem Size (elements)')
    ax2.set_ylabel('Throughput (GOPS)')
    ax2.set_title('Throughput vs Problem Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bandwidth vs Size
    ax3 = axes[1, 0]
    for algo in algorithms:
        algo_data = df[df['Algorithm'] == algo]
        ax3.semilogx(algo_data['Size'], algo_data['Bandwidth_GB_s'], 'o-', label=algo, linewidth=2)
    ax3.set_xlabel('Problem Size (elements)')
    ax3.set_ylabel('Memory Bandwidth (GB/s)')
    ax3.set_title('Memory Bandwidth vs Problem Size')
    ax3.axhline(y=504, color='red', linestyle='--', label='Theoretical Peak (504 GB/s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Efficiency comparison
    ax4 = axes[1, 1]
    largest_size_data = df[df['Size'] == df['Size'].max()]
    algorithms_subset = largest_size_data['Algorithm'].values
    throughputs = largest_size_data['Throughput_GOPS'].values
    
    bars = ax4.bar(algorithms_subset, throughputs, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax4.set_ylabel('Throughput (GOPS)')
    ax4.set_title(f'Algorithm Comparison (Size: {df["Size"].max():,} elements)')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, trends):
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.suptitle('CUDA Parallel Algorithms Performance Dashboard\nRTX 4070 Ti Super | Ada Lovelace Architecture', 
                 fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_dashboard.png', dpi=300, bbox_inches='tight')
    print("Advanced performance dashboard saved as 'performance_dashboard.png'")

def create_memory_analysis_plot(df):
    """Create memory-focused analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Memory Performance Analysis', fontsize=16)
    
    # Memory bandwidth vs problem size
    ax1 = axes[0, 0]
    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo]
        ax1.semilogx(algo_data['Size'], algo_data['Bandwidth_GB_s'], 'o-', label=algo, linewidth=2)
    
    theoretical_bw = 504  # RTX 4070 Ti Super
    ax1.axhline(y=theoretical_bw, color='red', linestyle='--', alpha=0.7, 
                label=f'Theoretical Peak ({theoretical_bw} GB/s)')
    ax1.set_xlabel('Problem Size (elements)')
    ax1.set_ylabel('Memory Bandwidth (GB/s)')
    ax1.set_title('Memory Bandwidth Scaling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bandwidth efficiency
    ax2 = axes[0, 1]
    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo]
        efficiency = (algo_data['Bandwidth_GB_s'] / theoretical_bw) * 100
        ax2.semilogx(algo_data['Size'], efficiency, 'o-', label=algo, linewidth=2)
    
    ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% Efficiency')
    ax2.set_xlabel('Problem Size (elements)')
    ax2.set_ylabel('Bandwidth Efficiency (%)')
    ax2.set_title('Memory Bandwidth Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Memory throughput vs compute throughput
    ax3 = axes[1, 0]
    for algo in df['Algorithm'].unique():
        algo_data = df[df['Algorithm'] == algo]
        ax3.scatter(algo_data['Bandwidth_GB_s'], algo_data['Throughput_GOPS'], 
                   label=algo, s=60, alpha=0.7)
    
    ax3.set_xlabel('Memory Bandwidth (GB/s)')
    ax3.set_ylabel('Compute Throughput (GOPS)')
    ax3.set_title('Memory vs Compute Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Algorithm classification
    ax4 = axes[1, 1]
    largest_data = df[df['Size'] == df['Size'].max()]
    
    # Classify algorithms as memory-bound or compute-bound
    classifications = []
    colors = []
    
    for _, row in largest_data.iterrows():
        bw_efficiency = (row['Bandwidth_GB_s'] / theoretical_bw) * 100
        if bw_efficiency > 70:
            classifications.append('Memory-Bound')
            colors.append('red')
        elif bw_efficiency > 40:
            classifications.append('Balanced')
            colors.append('orange')
        else:
            classifications.append('Compute-Bound')
            colors.append('green')
    
    bars = ax4.bar(largest_data['Algorithm'], largest_data['Bandwidth_GB_s'], color=colors)
    ax4.set_ylabel('Memory Bandwidth (GB/s)')
    ax4.set_title('Algorithm Classification')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add classification labels
    for i, (bar, classification) in enumerate(zip(bars, classifications)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 10,
                classification, ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('memory_analysis.png', dpi=300, bbox_inches='tight')
    print("Memory analysis plot saved as 'memory_analysis.png'")

def main():
    parser = argparse.ArgumentParser(description='Advanced CUDA performance visualization')
    parser.add_argument('input_file', help='Input CSV file with benchmark results')
    parser.add_argument('--dashboard', action='store_true', help='Create performance dashboard')
    parser.add_argument('--memory', action='store_true', help='Create memory analysis plots')
    parser.add_argument('--all', action='store_true', help='Create all visualizations')
    
    args = parser.parse_args()
    
    # Load data
    try:
        df = pd.read_csv(args.input_file)
        print(f"Loaded {len(df)} benchmark results")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    if args.all or args.dashboard:
        create_advanced_plots(df)
    
    if args.all or args.memory:
        create_memory_analysis_plot(df)
    
    if not (args.dashboard or args.memory or args.all):
        print("No visualization option selected. Use --help for options.")

if __name__ == '__main__':
    main()