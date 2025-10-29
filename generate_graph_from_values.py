import numpy as np
import matplotlib.pyplot as plt

# Values from the previous run
build_times = {
    'L2 Distance': 181.0,  # seconds
    'Cosine Similarity': 178.69  # seconds
}

load_times = {
    'L2 Distance': 1.13665,  # seconds
    'Cosine Similarity': 1.1280  # seconds
}

file_sizes = {
    'L2 Distance Index': 59.53,  # MB
    'L2 Distance Metadata': 1.39,  # MB
    'Cosine Similarity Index': 59.53,  # MB
    'Cosine Similarity Metadata': 1.39  # MB
}

# Breakdown times (estimated from percentages shown in pie charts)
breakdown_times = {
    'L2 Distance': {
        'Encode': 181 * 0.983,  # 98.3%
        'Index Build': 181 * 0.017  # 1%
    },
    'Cosine Similarity': {
        'Encode': 178.69 * 0.992,  # 99.2%
        'Index Build': 178.69 * 0.008  # 0.8%
    }
}

# Colors
color_l2 = '#FF1493'  # Deep Pink
color_cosine = '#8B00FF'  # Violet

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Index Building & Loading Performance Comparison', fontsize=16, fontweight='bold')

# Plot 1: Build times comparison (in seconds and minutes)
methods = ['L2 Distance', 'Cosine Similarity']
build_times_sec = [build_times['L2 Distance'], build_times['Cosine Similarity']]
build_times_min = [t/60 for t in build_times_sec]

colors = [color_l2, color_cosine]
bars = axes[0, 0].bar(methods, build_times_sec, color=colors, alpha=0.85, width=0.6)
axes[0, 0].set_ylabel('Time (seconds)', fontweight='bold', fontsize=12)
axes[0, 0].set_title('Total Index Building Time', fontweight='bold', fontsize=13)
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].set_ylim(0, max(build_times_sec) * 1.15)

# Add value labels
for bar, time_sec, time_min in zip(bars, build_times_sec, build_times_min):
    height = bar.get_height()
    label_text = f'{time_min:.2f}m\n({time_sec:.0f}s)'
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                   label_text,
                   ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 2: Load times comparison
methods_load = ['L2 Distance', 'Cosine Similarity']
load_times_vals = [load_times['L2 Distance'], load_times['Cosine Similarity']]

bars = axes[0, 1].bar(methods_load, load_times_vals, color=colors, alpha=0.85, width=0.6)
axes[0, 1].set_ylabel('Time (seconds)', fontweight='bold', fontsize=12)
axes[0, 1].set_title('Index Loading Time', fontweight='bold', fontsize=13)
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].set_ylim(0, max(load_times_vals) * 1.3)

# Add value labels
for bar, time in zip(bars, load_times_vals):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                   f'{time:.4f}s',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 3: File sizes comparison
file_names = ['L2 Index', 'L2 Metadata', 'Cosine Index', 'Cosine Metadata']
file_sizes_vals = [59.53, 1.39, 59.53, 1.39]
colors_files = [color_l2, color_l2, color_cosine, color_cosine]

bars = axes[1, 0].barh(file_names, file_sizes_vals, color=colors_files, alpha=0.85)
axes[1, 0].set_xlabel('File Size (MB)', fontweight='bold', fontsize=12)
axes[1, 0].set_title('Index and Metadata File Sizes', fontweight='bold', fontsize=13)
axes[1, 0].grid(axis='x', alpha=0.3)

# Add value labels
for bar, size in zip(bars, file_sizes_vals):
    width = bar.get_width()
    axes[1, 0].text(width, bar.get_y() + bar.get_height()/2.,
                   f'{size:.2f} MB',
                   ha='left', va='center', fontweight='bold', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Plot 4: Summary metrics
axes[1, 1].axis('off')
summary_text = "PERFORMANCE SUMMARY\n"
summary_text += "="*50 + "\n\n"

summary_text += f"L2 DISTANCE:\n"
summary_text += f"  Build Time: 3.02m (181s)\n"
summary_text += f"  Load Time: 1.1367s\n"
summary_text += f"  Total Size: 60.92 MB\n"
summary_text += f"  Breakdown:\n"
summary_text += f"    - Encode: 177.78s (98.3%)\n"
summary_text += f"    - Index Build: 3.07s (1.7%)\n\n"

summary_text += f"COSINE SIMILARITY:\n"
summary_text += f"  Build Time: 2.98m (179s)\n"
summary_text += f"  Load Time: 1.1280s\n"
summary_text += f"  Total Size: 60.92 MB\n"
summary_text += f"  Breakdown:\n"
summary_text += f"    - Encode: 177.35s (99.2%)\n"
summary_text += f"    - Index Build: 1.34s (0.8%)\n\n"

summary_text += f"COMPARISON:\n"
summary_text += f"  L2 vs Cosine Build: ~2.31s faster (Cosine)\n"
summary_text += f"  L2 vs Cosine Load: ~0.0087s faster (Cosine)\n"
summary_text += f"  File Sizes: Identical (60.92 MB each)"

axes[1, 1].text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='lightyellow', alpha=0.8, pad=1))

plt.tight_layout()

# Save the plot
output_file = 'index_building_times.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Graph saved as '{output_file}'")
print("\nPerformance Summary:")
print("="*50)
print(f"L2 Distance Build Time: 3.02m (181s)")
print(f"Cosine Similarity Build Time: 2.98m (179s)")
print(f"L2 Distance Load Time: 1.1367s")
print(f"Cosine Similarity Load Time: 1.1280s")
print(f"Both indexes: 60.92 MB total")
print("="*50)
