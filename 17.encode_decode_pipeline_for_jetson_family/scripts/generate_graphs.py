#!/usr/bin/env python3
"""
Performance Visualization Script for GPU vs CPU Pipeline Comparison
Creates comprehensive graphs from performance monitoring data
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
import argparse
from datetime import datetime
import numpy as np

def load_performance_data(csv_file):
    """Load performance data from CSV file"""
    try:
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        return df
    except Exception as e:
        print(f"Error loading CSV file {csv_file}: {e}")
        return None

def create_performance_comparison_graph(df, output_file="gpu_vs_cpu_power.png"):
    """Create comprehensive performance comparison graph"""
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('GPU vs CPU Pipeline Performance Comparison', fontsize=16, fontweight='bold')
    
    # Define colors for GPU and CPU
    gpu_color = '#2E8B57'  # Sea Green
    cpu_color = '#DC143C'  # Crimson
    
    # Filter data for GPU and CPU events
    gpu_events = df[df['event_name'].str.contains('GPU', na=False)]
    cpu_events = df[df['event_name'].str.contains('CPU', na=False)]
    
    # Plot 1: Power Consumption Over Time
    ax1 = axes[0, 0]
    if not gpu_events.empty:
        ax1.plot(gpu_events['timestamp'], gpu_events['power_consumption_watts'], 
                color=gpu_color, linewidth=2, label='GPU Pipeline', marker='o', markersize=3)
    if not cpu_events.empty:
        ax1.plot(cpu_events['timestamp'], cpu_events['power_consumption_watts'], 
                color=cpu_color, linewidth=2, label='CPU Pipeline', marker='s', markersize=3)
    
    ax1.set_title('Power Consumption Comparison', fontweight='bold')
    ax1.set_ylabel('Power (Watts)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    # Plot 2: GPU and CPU Usage
    ax2 = axes[0, 1]
    if not df.empty:
        ax2.plot(df['timestamp'], df['gpu_usage_percent'], 
                color=gpu_color, linewidth=2, label='GPU Usage (%)', alpha=0.8)
        ax2.plot(df['timestamp'], df['cpu_usage_percent'], 
                color=cpu_color, linewidth=2, label='CPU Usage (%)', alpha=0.8)
    
    ax2.set_title('GPU vs CPU Utilization', fontweight='bold')
    ax2.set_ylabel('Usage (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    # Plot 3: Memory Usage
    ax3 = axes[1, 0]
    if not df.empty:
        ax3.plot(df['timestamp'], df['gpu_memory_usage_mb'], 
                color=gpu_color, linewidth=2, label='GPU Memory (MB)', alpha=0.8)
        ax3.plot(df['timestamp'], df['total_memory_usage_mb'], 
                color=cpu_color, linewidth=2, label='System Memory (MB)', alpha=0.8)
    
    ax3.set_title('Memory Usage Comparison', fontweight='bold')
    ax3.set_ylabel('Memory (MB)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    # Plot 4: Temperature
    ax4 = axes[1, 1]
    if not df.empty:
        ax4.plot(df['timestamp'], df['temperature_celsius'], 
                color='#FF6347', linewidth=2, label='GPU Temperature', marker='d', markersize=3)
    
    ax4.set_title('Temperature Monitoring', fontweight='bold')
    ax4.set_ylabel('Temperature (°C)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    
    # Format x-axis for all subplots
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Performance comparison graph saved: {output_file}")
    
    return fig

def create_performance_summary_table(df):
    """Create a summary table of performance metrics"""
    
    # Separate GPU and CPU data
    gpu_data = df[df['event_name'].str.contains('GPU', na=False)]
    cpu_data = df[df['event_name'].str.contains('CPU', na=False)]
    
    summary = {
        'Metric': [
            'Average Power Consumption (W)',
            'Peak Power Consumption (W)',
            'Average GPU Usage (%)',
            'Average CPU Usage (%)',
            'Average Temperature (°C)',
            'Peak Temperature (°C)',
            'Average GPU Memory (MB)',
            'Peak GPU Memory (MB)'
        ],
        'GPU Pipeline': [
            gpu_data['power_consumption_watts'].mean() if not gpu_data.empty else 0,
            gpu_data['power_consumption_watts'].max() if not gpu_data.empty else 0,
            gpu_data['gpu_usage_percent'].mean() if not gpu_data.empty else 0,
            gpu_data['cpu_usage_percent'].mean() if not gpu_data.empty else 0,
            gpu_data['temperature_celsius'].mean() if not gpu_data.empty else 0,
            gpu_data['temperature_celsius'].max() if not gpu_data.empty else 0,
            gpu_data['gpu_memory_usage_mb'].mean() if not gpu_data.empty else 0,
            gpu_data['gpu_memory_usage_mb'].max() if not gpu_data.empty else 0
        ],
        'CPU Pipeline': [
            cpu_data['power_consumption_watts'].mean() if not cpu_data.empty else 0,
            cpu_data['power_consumption_watts'].max() if not cpu_data.empty else 0,
            cpu_data['gpu_usage_percent'].mean() if not cpu_data.empty else 0,
            cpu_data['cpu_usage_percent'].mean() if not cpu_data.empty else 0,
            cpu_data['temperature_celsius'].mean() if not cpu_data.empty else 0,
            cpu_data['temperature_celsius'].max() if not cpu_data.empty else 0,
            cpu_data['gpu_memory_usage_mb'].mean() if not cpu_data.empty else 0,
            cpu_data['gpu_memory_usage_mb'].max() if not cpu_data.empty else 0
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    
    # Round numerical values
    summary_df['GPU Pipeline'] = summary_df['GPU Pipeline'].round(2)
    summary_df['CPU Pipeline'] = summary_df['CPU Pipeline'].round(2)
    
    return summary_df

def create_efficiency_analysis(df):
    """Create efficiency analysis comparing GPU vs CPU performance"""
    
    gpu_data = df[df['event_name'].str.contains('GPU', na=False)]
    cpu_data = df[df['event_name'].str.contains('CPU', na=False)]
    
    if gpu_data.empty or cpu_data.empty:
        print("Warning: Missing GPU or CPU data for efficiency analysis")
        return None
    
    # Calculate efficiency metrics
    gpu_avg_power = gpu_data['power_consumption_watts'].mean()
    cpu_avg_power = cpu_data['power_consumption_watts'].mean()
    
    gpu_avg_usage = gpu_data['gpu_usage_percent'].mean()
    cpu_avg_usage = cpu_data['cpu_usage_percent'].mean()
    
    # Power efficiency (lower is better)
    power_efficiency_ratio = gpu_avg_power / cpu_avg_power if cpu_avg_power > 0 else float('inf')
    
    # Processing efficiency (higher GPU usage typically means better parallelization)
    processing_efficiency = gpu_avg_usage / cpu_avg_usage if cpu_avg_usage > 0 else float('inf')
    
    analysis = {
        'GPU Average Power (W)': gpu_avg_power,
        'CPU Average Power (W)': cpu_avg_power,
        'Power Efficiency Ratio (GPU/CPU)': power_efficiency_ratio,
        'GPU Average Usage (%)': gpu_avg_usage,
        'CPU Average Usage (%)': cpu_avg_usage,
        'Processing Efficiency Ratio': processing_efficiency,
        'Recommendation': 'GPU' if power_efficiency_ratio < 1.5 and processing_efficiency > 1.2 else 'CPU'
    }
    
    return analysis

def print_performance_report(df, summary_df, efficiency_analysis):
    """Print comprehensive performance report"""
    
    print("\n" + "="*60)
    print("           GPU vs CPU PIPELINE PERFORMANCE REPORT")
    print("="*60)
    
    print("\n PERFORMANCE SUMMARY TABLE")
    print("-" * 60)
    print(summary_df.to_string(index=False))
    
    if efficiency_analysis:
        print("\n⚡ EFFICIENCY ANALYSIS")
        print("-" * 60)
        for key, value in efficiency_analysis.items():
            if isinstance(value, float):
                print(f"{key:.<40} {value:.2f}")
            else:
                print(f"{key:.<40} {value}")
        
        print(f"\n RECOMMENDATION: {efficiency_analysis['Recommendation']} Pipeline")
        
        if efficiency_analysis['Recommendation'] == 'GPU':
            print("    GPU pipeline shows better performance characteristics")
            print("    Recommended for: High-throughput video processing")
        else:
            print("    CPU pipeline shows better performance characteristics") 
            print("    Recommended for: General-purpose video processing")
    
    # Event timeline
    events = df[df['event_name'].notna() & (df['event_name'] != '')]
    if not events.empty:
        print("\n EVENT TIMELINE")
        print("-" * 60)
        for _, event in events.iterrows():
            timestamp = event['timestamp'].strftime('%H:%M:%S')
            print(f"{timestamp} - {event['event_name']}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Generate performance comparison graphs')
    parser.add_argument('csv_file', help='Path to performance CSV file')
    parser.add_argument('--output', '-o', default='gpu_vs_cpu_power.png', 
                       help='Output graph filename (default: gpu_vs_cpu_power.png)')
    parser.add_argument('--report', '-r', action='store_true',
                       help='Generate detailed performance report')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading performance data from: {args.csv_file}")
    df = load_performance_data(args.csv_file)
    
    if df is None:
        print("Failed to load performance data")
        return 1
    
    if df.empty:
        print("No performance data found in CSV file")
        return 1
    
    print(f"Loaded {len(df)} data points")
    
    # Create performance graphs
    print("Generating performance comparison graphs...")
    try:
        fig = create_performance_comparison_graph(df, args.output)
        plt.show() if '--show' in sys.argv else plt.close(fig)
    except Exception as e:
        print(f"Error creating graphs: {e}")
        return 1
    
    # Generate summary and analysis
    summary_df = create_performance_summary_table(df)
    efficiency_analysis = create_efficiency_analysis(df)
    
    # Print report if requested
    if args.report or len(sys.argv) == 2:  # Default to showing report
        print_performance_report(df, summary_df, efficiency_analysis)
    
    # Save summary to file
    summary_file = args.csv_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("GPU vs CPU Pipeline Performance Summary\n")
        f.write("="*50 + "\n\n")
        f.write("Performance Summary Table:\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")
        
        if efficiency_analysis:
            f.write("Efficiency Analysis:\n")
            for key, value in efficiency_analysis.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
    
    print(f"\n Files generated:")
    print(f"    Graph: {args.output}")
    print(f"    Summary: {summary_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())