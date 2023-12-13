#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from utils.energy_calc import compute_grav_potential_energy, compute_elec_potential_energy
import cmocean
matplotlib.use('Agg')


"""Plot positions
simulation: simulation w pos vel + energies for time interval
dt: time step
output_dir: output directory for plot pngs
skip_frames: plot every nth time step 
"""
def render_positions(simulation, dt, output_dir, skip_frames=None):

    # skip every nth frame if defined
    if skip_frames:
        if dt % skip_frames != 0:
            return
    
    fig, ax = plt.subplots()

    x_range = np.ptp(simulation.vec_store[:, :, 0])
    y_range = np.ptp(simulation.vec_store[:, :, 1])
    max_range = max(x_range, y_range)

    xmin, xmax = np.min(simulation.vec_store[:, :, 0])-0.1*max_range, np.max(simulation.vec_store[:, :, 0])+0.1*max_range
    ymin, ymax = np.min(simulation.vec_store[:, :, 1])-0.1*max_range, np.max(simulation.vec_store[:, :, 1])+0.1*max_range

    # plot positions x & y positions for each body
    for i in range(simulation.vec_store.shape[1]):
        ax.plot(simulation.vec_store[dt, i, 0], simulation.vec_store[dt, i, 1], 'o')

    ax.set_title(f'Time step: {dt}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    plt.savefig(os.path.join(output_dir, f'timestep_{dt}.png'))
    plt.close(fig)


"""Plot positions w potential
simulation: simulation w pos vel + energies for time interval
dt: time step
output_dir: output directory for plot pngs
vmin: min value for colorbar
vmax: max value for colorbar
skip_frames: plot every nth time step 
"""
def render_positions_w_potential(simulation, dt, output_dir, vmin, vmax, skip_frames=None):

    # skip every nth frame if defined
    if skip_frames:
        if dt % skip_frames != 0:
            return
    
    fig, ax = plt.subplots()

    x_range = np.ptp(simulation.vec_store[:, :, 0])
    y_range = np.ptp(simulation.vec_store[:, :, 1])
    max_range = max(x_range, y_range)

    xmin, xmax = np.min(simulation.vec_store[:, :, 0])-0.1*max_range, np.max(simulation.vec_store[:, :, 0])+0.1*max_range
    ymin, ymax = np.min(simulation.vec_store[:, :, 1])-0.1*max_range, np.max(simulation.vec_store[:, :, 1])+0.1*max_range

    # plot positions x & y positions for each body
    for i in range(simulation.vec_store.shape[1]):
        ax.plot(simulation.vec_store[dt, i, 0], simulation.vec_store[dt, i, 1], 'o', label=f'Body {i + 1}', markeredgecolor='black', fillstyle='none')

    ax.set_title(f'Time step: {dt}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_aspect('equal')

    # generate array for potential gradient
    x_range = np.linspace(xmin, xmax, 100)
    y_range = np.linspace(ymin, ymax, 100)

    X, Y = np.meshgrid(x_range, y_range)

    potential_grid = np.zeros_like(X)

    if simulation.forces == "gravity":

        # calculate and plot potential energy for each grid space
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                position = np.array([X[i, j], Y[i, j], 0])
                for n in range(simulation.N):
                    potential_grid[i, j] += compute_grav_potential_energy(simulation.vec_store[dt, n, :3], simulation.masses[n], position)

        pcm = ax.pcolormesh(X, Y, np.log10(potential_grid), cmap='viridis', shading='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=r"$\log_{10}\phi$")
    
    elif simulation.forces == "coulomb":

        cmap = cmocean.cm.matter_r

        # calculate and plot potential energy for each grid space
        for i in range(len(x_range)):
            for j in range(len(y_range)):
                position = np.array([X[i, j], Y[i, j], 0])
                for n in range(simulation.N):
                    potential_grid[i, j] += compute_elec_potential_energy(simulation.vec_store[dt, n, :3], simulation.charges[n], position)

        pcm = ax.pcolormesh(X, Y, potential_grid, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(pcm, ax=ax, label=r"$\phi$")

    plt.savefig(os.path.join(output_dir, f'timestep_{dt}.png'))
    plt.close(fig)


"""Plot orbit/trajectories
simulation: simulation w pos vel + energies for time interval
save_path: file name for plot png
"""
def plot_orbits(simulation, save_path="orbits"):

    fig, ax = plt.subplots()

    # plot all positions for each body
    positions = simulation.vec_store[:, :, :3]
    for i in range(positions.shape[1]):
        ax.scatter(positions[:, i, 0], positions[:, i, 1], s=1,color=[.7,.7,1])
        ax.scatter(positions[-1, i, 0],positions[-1, i, 1],s=10,color='blue')

    ax.set_title('Orbits')
    ax.set_xlabel('X Position')  
    ax.set_ylabel('Y Position')

    plt.savefig(f"{save_path}.png")


"""Plot energy
simulation: simulation w pos vel + energies for time interval
save_path: file name for plot png
"""
def plot_energy(simulation, save_path="energy"):

    fig, ax = plt.subplots()

    timesteps = simulation.time_steps

    total_energy = simulation.energy_store

    KE = simulation.KE_store
    U = simulation.U_store

    # plot energy of system at each time step
    ax.plot(timesteps, total_energy, label="Total Energy")
    ax.plot(timesteps, KE, label="Kinetic Energy")
    ax.plot(timesteps, U, label="Potential Energy")

    plt.savefig(f"{save_path}.png")