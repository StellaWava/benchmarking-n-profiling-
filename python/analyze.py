import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import os


# csv_path = "roofline.csv"
# if not os.path.exists(csv_path):
#     csv_path = '../roofline.csv'


# try:
#     df = pd.read_csv(csv_path)
# except Exception as e:
#     print(f"Error reading CSV: {e}")


# df.columns = df.columns.str.strip()

# #filter data
# cpu_data = df[df['threads'] != 'CUDA']
# gpu_data = df[df['threads'] == 'CUDA']

# best_cpu = cpu_data.loc[cpu_data['performance (GFLOP/s)'].idxmax()] if not cpu_data.empty else None
# best_gpu = gpu_data.loc[gpu_data['performance (GFLOP/s)'].idxmax()] if not gpu_data.empty else None


# # 3. Setup the Plot
# plt.figure(figsize=(10, 7))

# # Plot data points
# if best_cpu is not None:
#     plt.scatter(best_cpu['AI'], best_cpu['performance (GFLOP/s)'], 
#                 color='tab:blue', s=150, zorder=5, label=f"CPU Peak ({best_cpu['threads']} threads)")
#     plt.text(best_cpu['AI']*1.1, best_cpu['performance (GFLOP/s)'], 
#              f"{best_cpu['performance (GFLOP/s)']:.2f} GFLOP/s\n({best_cpu['bandwidth (GB/s)']:.1f} GB/s)", 
#              va='center', color='tab:blue', fontweight='bold')

# if best_gpu is not None:
#     plt.scatter(best_gpu['AI'], best_gpu['performance (GFLOP/s)'], 
#                 color='tab:orange', s=150, zorder=5, label="GPU Peak (CUDA)")
#     plt.text(best_gpu['AI']*1.1, best_gpu['performance (GFLOP/s)'], 
#              f"{best_gpu['performance (GFLOP/s)']:.2f} GFLOP/s\n({best_gpu['bandwidth (GB/s)']:.1f} GB/s)", 
#              va='center', color='tab:orange', fontweight='bold')

# # 4. Chart configuration (Log-Log Scale is standard for Roofline)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Arithmetic Intensity (FLOP/byte)', fontsize=12)
# plt.ylabel('Performance (GFLOP/s)', fontsize=12)
# plt.title('Roofline Model - Initial Baseline (Stream Triad)', fontsize=14, pad=15)
# plt.grid(True, which="both", ls="--", alpha=0.5)

# # Adjust axes limits to leave room for future compute-bound dwarfs
# plt.xlim(0.01, 100.0)
# plt.ylim(0.1, 10000.0)
# plt.legend(loc='upper left', fontsize=11)

# # Save the plot
# output_png = 'roofline_chart.png'
# plt.savefig(output_png, dpi=300, bbox_inches='tight')
# print(f"Roofline chart successfully generated and saved to '{output_png}'")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Resolve exact CSV paths from your framework structure
csv_path = "roofline.csv"
if not os.path.exists(csv_path):
    csv_path = '../roofline.csv'

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Could not locate roofline data file at path: {csv_path}")

# 2. Read dataset and sanitize any trailing whitespace out of column labels
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# 3. Configure Figure Window on log-log axis bounds
plt.figure(figsize=(11, 7.5))
ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')

# Generate a clean arithmetic intensity evaluation range for drawing the lines
ai_space = np.logspace(-3, 3, 1000)

# 4. Group execution points by your exact column key: 'dwarf name'
dwarf_groups = df.groupby('dwarf name')

# Colors and marker pools to differentiate your models clearly
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']

for i, (name, group) in enumerate(dwarf_groups):
    color = colors[i % len(colors)]
    marker = markers[i % len(markers)]
    
    # Extract the custom boundaries logged for this specific model configuration
    peak_compute = group['peak_compute(GFLOP/s)'].max()
    peak_bandwidth = group['peak_bandwidth(GB/s)'].max()
    ridge_point = group['ridge_point'].max()
    
    # Handle optional zero/null case if empty
    if pd.isna(ridge_point) or ridge_point == 0:
        ridge_point = peak_compute / peak_bandwidth

    # 5. Draw the dedicated Roofline boundaries for this specific model
    roofline_curve = np.minimum(peak_compute, peak_bandwidth * ai_space)
    plt.plot(ai_space, roofline_curve, color=color, linestyle='-', linewidth=2, 
             label=f'{name} Roof (BW: {peak_bandwidth:.0f} GB/s, Peak: {peak_compute:.0f} GFLOP/s)')
    
    # 6. Draw the dedicated vertical Ridge Line for this specific model
    plt.axvline(x=ridge_point, color=color, linestyle=':', alpha=0.6,
                label=f'{name} Ridge Point ({ridge_point:.2f} FLOP/byte)')
    
    # Sort group by problem_size to show sequential sweep increase curves correctly
    sorted_group = group.sort_values(by='problem_size')
    
    # 7. Scatter plot the actual empirical dots for the sweeps
    plt.scatter(
        sorted_group['AI'], 
        sorted_group['measured_performance(GFLOP/s)'], 
        color=color, 
        s=100, 
        edgecolors='black', 
        marker=marker, 
        zorder=5
    )
    
    # Draw an explicit trailing path line tracing the sweep increase steps
    plt.plot(
        sorted_group['AI'], 
        sorted_group['measured_performance(GFLOP/s)'], 
        color=color, 
        linestyle='--', 
        alpha=0.5,
        zorder=4
    )
    
    # Annotate problem sizes directly next to the scatter sweep dots
    for _, row in sorted_group.iterrows():
        plt.annotate(
            f"N={int(row['problem_size'])}",
            (row['AI'], row['measured_performance(GFLOP/s)']),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            fontweight='bold',
            color=color
        )

# 8. Add clear region markers and structural labels
machine_label = df['machine'].iloc[0] if 'machine' in df.columns else "GPU Backend"
plt.title(f"Roofline Analysis Sweep Performance Map\nTarget Machine: {machine_label}", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Arithmetic Intensity (AI) (FLOP/byte)", fontsize=12, fontweight='semibold')
plt.ylabel("Measured Performance (GFLOP/s)", fontsize=12, fontweight='semibold')

# Label raw architectural region orientations safely based on graph layout rules
plt.text(0.002, peak_compute * 0.1, "Memory Bound)", 
         fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
plt.text(30.0, peak_compute * 0.7, "Compute Bound)", 
         fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

# Final framing cleanups
plt.xlim(0.001, 1000.0)
plt.ylim(0.1, df['peak_compute(GFLOP/s)'].max() * 3.0)
plt.grid(True, which="both", linestyle=":", alpha=0.35)
plt.legend(loc='upper left', fontsize=8, framealpha=0.95, edgecolor='gray')
plt.tight_layout()

# Save finalized performance asset back to your workspace directory
plt.savefig("gpu_roofline_plot.png", dpi=300)
plt.show()

print("Plotting sequence complete. 'gpu_sweeps_roofline_plot.png' compiled successfully.")