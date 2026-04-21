import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# 1. Prepare data (order: None → Wavelet → LogGabor → Fusion; metrics: Accuracy, Precision, Recall, F1-Score, IoU)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'IoU']  # X-axis: Metrics
branches = ['None', 'Wavelet', 'LogGabor', 'Fusion']  # Updated branch order
# Metric values (converted to % by ×100; row=branch, column=metric; order matches branches/metrics)
data = np.array([
    [0.9814, 0.8713, 0.6703, 0.7577, 0.6100],  # None (1st)
    [0.9836, 0.8724, 0.7265, 0.7927, 0.6567],  # Wavelet (2nd)
    [0.9833, 0.8787, 0.7122, 0.7867, 0.6484],  # LogGabor (3rd, updated name)
    [0.9832, 0.7646, 0.8841, 0.8201, 0.6950]   # Fusion (4th)
]) * 100  # Convert to percentage (核心：小数→百分比)

# 2. Figure configuration
fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.18  # Suitable for 4 branches per metric
x = np.arange(len(metrics))  # X-axis positions (one per metric)
# Professional color palette (consistent with branch order)
colors = ['#6A994E', '#2E86AB', '#A23B72', '#F18F01']

# 3. Plot: Each branch → color; each metric → 4 bars
for i, (branch, color) in enumerate(zip(branches, colors)):
    bars = ax.bar(x + i * bar_width, data[i, :], width=bar_width, label=branch, color=color, alpha=0.8)
    
# 4. Figure beautification
ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric Value (%)', fontsize=12, fontweight='bold')  # Y-label updated to %

# X-axis ticks (center-aligned for 4 bars per metric)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(metrics, ha='center', fontsize=10)

# Y-axis range (adapted to percentage: 55% → 102% to highlight differences)
ax.set_ylim(0, 102)

# Add horizontal grid lines (enhance readability)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Legend (matches updated branch order)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Adjust layout to avoid label cutoff
plt.tight_layout()

# 5. Save and display
plt.savefig('branch_models_comparison_percentage.png', dpi=300, bbox_inches='tight')
plt.show()