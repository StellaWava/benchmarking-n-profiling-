import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import os


csv_path = "roofline.csv"
if not os.path.exists(csv_path):
    csv_path = '../roofline.csv'


try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error reading CSV: {e}")


df.columns = df.columns.str.strip()

#filter data
cpu_data = df[df['threads'] != 'CUDA']
gpu_data = df[df['threads'] == 'CUDA']

best_cpu = cpu_data.loc[cpu_data['performance (GFLOP/s)'].idxmax()] if not cpu_data.empty else None
best_gpu = gpu_data.loc[gpu_data['performance (GFLOP/s)'].idxmax()] if not gpu_data.empty else None


# 3. Setup the Plot
plt.figure(figsize=(10, 7))

# Plot data points
if best_cpu is not None:
    plt.scatter(best_cpu['AI'], best_cpu['performance (GFLOP/s)'], 
                color='tab:blue', s=150, zorder=5, label=f"CPU Peak ({best_cpu['threads']} threads)")
    plt.text(best_cpu['AI']*1.1, best_cpu['performance (GFLOP/s)'], 
             f"{best_cpu['performance (GFLOP/s)']:.2f} GFLOP/s\n({best_cpu['bandwidth (GB/s)']:.1f} GB/s)", 
             va='center', color='tab:blue', fontweight='bold')

if best_gpu is not None:
    plt.scatter(best_gpu['AI'], best_gpu['performance (GFLOP/s)'], 
                color='tab:orange', s=150, zorder=5, label="GPU Peak (CUDA)")
    plt.text(best_gpu['AI']*1.1, best_gpu['performance (GFLOP/s)'], 
             f"{best_gpu['performance (GFLOP/s)']:.2f} GFLOP/s\n({best_gpu['bandwidth (GB/s)']:.1f} GB/s)", 
             va='center', color='tab:orange', fontweight='bold')

# 4. Chart configuration (Log-Log Scale is standard for Roofline)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
plt.ylabel('Performance (GFLOP/s)', fontsize=12)
plt.title('Roofline Model - Initial Baseline (Stream Triad)', fontsize=14, pad=15)
plt.grid(True, which="both", ls="--", alpha=0.5)

# Adjust axes limits to leave room for future compute-bound dwarfs
plt.xlim(0.01, 100.0)
plt.ylim(0.1, 10000.0)
plt.legend(loc='upper left', fontsize=11)

# Save the plot
output_png = 'roofline_chart.png'
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"Roofline chart successfully generated and saved to '{output_png}'")
