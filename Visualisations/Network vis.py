import matplotlib.pyplot as plt
import numpy as np

# Define the structure of the NN
layers = [4, 4, 4, 2]
layer_names = [
    ['Mass Number (A)', 'Atomic Number (Z)', r'Pairing Term ($\delta$)', 'Q_beta- WS4 (keV)'],
    ['', '', '', ''],
    ['', '', '', ''],
    [r'$\mu$', r'$\sigma$']
]

# Colors for each layer
colors = ['lightgrey', '#FF4444', '#FF4444', 'grey']

# Create a figure
fig, ax = plt.subplots(figsize=(16, 9))

# Calculate positions of nodes
layer_spacing = 1.0 / (len(layers) + 3)
node_spacing = [0.2, 0.2, 0.2, 0.35]
x_positions = np.cumsum([0.2] + [layer_spacing] * (len(layers) - 1))
positions = {}

for layer_idx, (layer, spacing) in enumerate(zip(layers, node_spacing)):
    y_positions = np.linspace(1, 0, layer)
    positions[layer_idx] = list(zip([x_positions[layer_idx]] * layer, y_positions))

# Draw nodes and labels
for i, (num_nodes, color) in enumerate(zip(layers, colors)):
    for j, pos in enumerate(positions[i]):
        ax.plot(pos[0], pos[1], 'o', markersize=40, color=color, zorder=3)
        if layer_names[i][j]:
            if i == len(layers) - 1:
                ax.text(pos[0] + 0.05, pos[1], layer_names[i][j], fontsize=36, ha='left', va='center')
            else:
                ax.text(pos[0] - 0.05, pos[1], layer_names[i][j], fontsize=36, ha='right', va='center')

        if i in [1, 2] and j == 1:
            vertical_dot_y_positions = np.linspace(positions[i][j][1], positions[i][j+1][1], 4)[1:3]
            for dot_y in vertical_dot_y_positions:
                ax.plot(pos[0], dot_y, '.', color='black', markersize=8, zorder=3)

# Draw connections
for i in range(len(layers) - 1):
    for start_pos in positions[i]:
        for end_pos in positions[i + 1]:
            linewidth = np.random.uniform(0.5, 2.5)
            ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 'k-', linewidth=linewidth, zorder=1)

# Adjust the plot
ax.set_xlim(-0.1, x_positions[-1] + 0.1)
ax.set_ylim(-0.2, 1.1)
ax.axis('off')  # Turn off the axis

plt.savefig('Visualisations/NN.png', dpi=900, transparent=True)
plt.show()