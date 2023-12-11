#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from utils.energy_calc import compute_grav_potential_energy
import matplotlib.colors as colors
matplotlib.use('Agg')

def render_positions(simulation, dt, output_dir):
    
    fig, ax = plt.subplots()

    x_range = np.ptp(simulation.vec_store[:, :, 0])
    y_range = np.ptp(simulation.vec_store[:, :, 1])

    max_range = max(x_range, y_range)

    xmin, xmax = np.min(simulation.vec_store[:, :, 0])-0.1*max_range, np.max(simulation.vec_store[:, :, 0])+0.1*max_range
    ymin, ymax = np.min(simulation.vec_store[:, :, 1])-0.1*max_range, np.max(simulation.vec_store[:, :, 1])+0.1*max_range

    for i in range(simulation.vec_store.shape[1]):
        ax.plot(simulation.vec_store[dt, i, 0], simulation.vec_store[dt, i, 1], 'o')

    ax.set_title(f'Time step: {dt}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.legend()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.savefig(os.path.join(output_dir, f'timestep_{dt}.png'))
    plt.close(fig)

def render_positions_w_potential(simulation, dt, output_dir, vmin, vmax):
    
    fig, ax = plt.subplots()

    x_range = np.ptp(simulation.vec_store[:, :, 0])
    y_range = np.ptp(simulation.vec_store[:, :, 1])

    max_range = max(x_range, y_range)

    xmin, xmax = np.min(simulation.vec_store[:, :, 0])-0.1*max_range, np.max(simulation.vec_store[:, :, 0])+0.1*max_range
    ymin, ymax = np.min(simulation.vec_store[:, :, 1])-0.1*max_range, np.max(simulation.vec_store[:, :, 1])+0.1*max_range

    for i in range(simulation.vec_store.shape[1]):
        ax.plot(simulation.vec_store[dt, i, 0], simulation.vec_store[dt, i, 1], 'o', label=f'Body {i + 1}', markeredgecolor='black', fillstyle='none')

    ax.set_title(f'Time step: {dt}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # ax.legend()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect('equal')

    # potential gradient
    x_range = np.linspace(xmin, xmax, 100)
    y_range = np.linspace(ymin, ymax, 100)

    X, Y = np.meshgrid(x_range, y_range)

    potential_grid = np.zeros_like(X)

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            position = np.array([X[i, j], Y[i, j], 0])
            for n in range(simulation.N):
                potential_grid[i, j] += compute_grav_potential_energy(simulation.vec_store[dt, n, :3], simulation.masses[n], position)

    pcm = ax.pcolormesh(X, Y, np.log10(potential_grid), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
    fig.colorbar(pcm, ax=ax, label=r"$\log_{10}\phi$")

    plt.savefig(os.path.join(output_dir, f'timestep_{dt}.png'))
    plt.close(fig)