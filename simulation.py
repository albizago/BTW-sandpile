# -*- coding: utf-8 -*-
"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SIMULATION OF BTW SANDPILE MODEL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Python 3.12

@author: Alberto Zaghini

ColSup 2024
"""

# Packages needed
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anmt

# Some efficiency settings
plt.style.use('fast')
sys.setrecursionlimit(3000)
plt.ion()

# Default parameters values

# border_conditions = False
# directed = False
# grid_size = 7
# sim_time = 300
# n_added = 1

# Setting toppling threshold
threshold = 4

# Extraction of parameters from configuration file
border_conditions, directed, grid_size, sim_time, n_added = np.loadtxt(
    'params.txt', dtype='int')

# Random number generator initialization
rangen = np.random.default_rng(seed=42)

# Initialization of position and height grids
sandpile_grid = np.indices((grid_size, grid_size))
sandpile_heights = np.zeros((grid_size, grid_size))

# Initialization of auxiliary numerical variables
x, y, height = 0, 0, 0

# Array for avalanches' data initialization
avalanches = np.full((0, 4*grid_size, 2), grid_size, dtype=int)

# Matplotlib figure
plt.close()
fig, grid = plt.subplots()
fig.set_figheight(8)
fig.set_figwidth(8)
grid.invert_yaxis()
grid.set_axis_off()
# ax = fig.add_subplot(111, projection='3d')
# ax.disable_mouse_rotation()


# Function to perform (recursively) topplings
def topple(x=[0, 0], idx=0, pt=0):
    global avalanches, sandpile_heights
    i, j = x
    # Height before toppling is stored
    height = sandpile_heights[i, j]
    # The sand drops. This is done before propagating the avalanche
    # in order to avoid paradoxical situations
    sandpile_heights[i, j] -= (threshold+1)
    # Not directed = no gravity, sand can fall upwards
    if directed == False:
        # Periodic border conditions (clearly at most in one direction)
        # allow sand to fall to the other side of the grid
        if i > 0 or border_conditions == True:
            # Sand adds up
            sandpile_heights[i-1, j] += 1
            # If the resulting height is supercritical,
            # the avalanche propagates
            if sandpile_heights[i-1, j] > threshold:
                avalanches[idx, pt] = [i-1, j]
                # The index in the avalanche array is incremented
                pt += 1
                # Recursion
                pt = topple([i-1, j], idx, pt)
        if i < grid_size-1 or border_conditions == True:
            sandpile_heights[(i+1) % grid_size, j] += 1
            if sandpile_heights[(i+1) % grid_size, j] > threshold:
                avalanches[idx, pt] = [(i+1) % grid_size, j]
                pt += 1
                pt = topple([(i+1) % grid_size, j], idx, pt)
        if j > 0:
            sandpile_heights[i, j-1] += 1
            if sandpile_heights[i, j-1] > threshold:
                avalanches[idx, pt] = [i, j-1]
                pt += 1
                pt = topple([i, j-1], idx, pt)
        if j < grid_size-1:
            sandpile_heights[i, j+1] += 1
            if sandpile_heights[i, j+1] > threshold:
                avalanches[idx, pt] = [i, j+1]
                pt += 1
                pt = topple([i, j+1], idx, pt)
    # Directed: only downwards
    elif directed == True:
        if sandpile_heights[i-1, j] < height and (i > 0 or border_conditions == True):
            sandpile_heights[i-1, j] += 1
            if sandpile_heights[i-1, j] > threshold:
                avalanches[idx, pt] = [i-1, j]
                pt += 1
                pt = topple([i-1, j], idx, pt)
        if sandpile_heights[(i+1) % grid_size, j] < height and (i < grid_size-1 or border_conditions == True):
            sandpile_heights[(i+1) % grid_size, j] += 1
            if sandpile_heights[(i+1) % grid_size, j] > threshold:
                avalanches[idx, pt] = [(i+1) % grid_size, j]
                pt += 1
                pt = topple([(i+1) % grid_size, j], idx, pt)
        if j > 0 and sandpile_heights[i, j-1] < height:
            sandpile_heights[i, j-1] += 1
            if sandpile_heights[i, j-1] > threshold:
                avalanches[idx, pt] = [i, j-1]
                pt += 1
                pt = topple([i, j-1], idx, pt)
        if j < grid_size-1 and sandpile_heights[i, j+1] < height:
            sandpile_heights[i, j+1] += 1
            if sandpile_heights[i, j+1] > threshold:
                avalanches[idx, pt] = [i, j+1]
                pt += 1
                pt = topple([i, j+1], idx, pt)
    return pt


mesh = grid.pcolormesh(sandpile_grid[1], sandpile_grid[0],
                       sandpile_heights, vmin=0.0, vmax=5.0, shading='nearest')
# barchart = ax.bar3d(sandpile_grid[0].flatten(), sandpile_grid[1].flatten(
# ), np.zeros(grid_size**2), .9, .9, sandpile_heights.flatten(), shade=False, color='darkgoldenrod')

# Global index for recorded avalanches
idx = 0

# Function to perform animation


def update(frame):
    global avalanches, newarray, sandpile_heights, new, idx
    # Extract cells with supercritical height
    excess = np.transpose(np.asarray(sandpile_heights > threshold).nonzero())
    k = np.size(excess, axis=0)
    if k > 0:
        # Initialize storage for avalanches' data
        avalanches = np.pad(avalanches, ((0, k), (0, 0), (0, 0)),
                            'constant', constant_values=grid_size)
        for u in range(np.size(excess, axis=0)):
            # start avalanche
            avalanches[idx][0] = excess[u]
            topple(excess[u], idx, 1)
            idx += 1
    else:
        # Add new grains at random
        """
        for k in range(n_added):
            x = rangen.integers(0, grid_size)
            y = rangen.integers(0, grid_size)
            sandpile_heights[x, y] += 1
        """
        sandpile_heights[int(grid_size/2), int(grid_size/2)] += n_added

    # Update figure
    mesh.set_array(sandpile_heights)
    # barchart = ax.bar3d(sandpile_grid[0].flatten(), sandpile_grid[1].flatten(
    # ), np.zeros(grid_size**2), .9, .9, sandpile_heights.flatten())


# Animation
writer = anmt.PillowWriter(fps=15)
ani = anmt.FuncAnimation(fig=fig, func=update,
                         frames=sim_time*2, blit=False, repeat=False)
# Export to animated image
name = datetime.datetime.now().strftime(" %b_%d - %H_%M_%S")
ani.save('./sim/sim-' + str(name) + '.gif', writer=writer)
plt.show()
# plt.close(fig)

# Save avalanches' data
np.savez('data', avalanches=avalanches)

# Final message
print('Execution completed')
