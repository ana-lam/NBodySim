#!/usr/bin/env python3

import os
from natsort import natsorted
import moviepy.video.io.ImageSequenceClip
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from utils.energy_calc import compute_grav_potential_energy
from utils.visualization import render_positions, render_positions_w_potential


"""Create animation
simulation: simulation w pos vel + energies for time interval
output_dir: output directory for plot pngs
file_name: mp4 file name 
potential_on: display potential energy gradient
fps: frame rate per second
skip_frames: plot every nth time step 
"""
def create_animation(simulation, output_dir, file_name="output", potential_on=False, fps=30, skip_frames=None):

    plt.close('all')

    # do one potential energy grid calculation to determine vmin and vmax for colorbar
    x_range = np.ptp(simulation.vec_store[:, :, 0])
    y_range = np.ptp(simulation.vec_store[:, :, 1])
    max_range = max(x_range, y_range)

    xmin, xmax = np.min(simulation.vec_store[:, :, 0])-0.1*max_range, np.max(simulation.vec_store[:, :, 0])+0.1*max_range
    ymin, ymax = np.min(simulation.vec_store[:, :, 1])-0.1*max_range, np.max(simulation.vec_store[:, :, 1])+0.1*max_range
    
    x_range = np.linspace(xmin, xmax, 100)
    y_range = np.linspace(ymin, ymax, 100)

    X, Y = np.meshgrid(x_range, y_range)
    potential_grid = np.zeros_like(X)

    for i in range(len(x_range)):
        for j in range(len(y_range)):    
            position = np.array([X[i, j], Y[i, j], 0])
            for n in range(simulation.N):
                potential_grid[i, j] += compute_grav_potential_energy(simulation.vec_store[1, n, :3], simulation.masses[n], position)
    
    # determine vmax and vmin with buffer
    vmax = np.max(np.log10(potential_grid))
    vmin = np.min(np.log10(potential_grid))
    v_range = vmax-vmin
    vmin = vmin - 0.1*v_range
    vmax = vmax - 0.1*v_range

    # make output directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # set up multiprocessing by batching time steps into # of CPUs
    n_cpu = mp.cpu_count()
    pool = mp.Pool(processes=n_cpu)

    simulations_batch = np.array_split(simulation.vec_store, n_cpu)
    offset = len(simulations_batch[0])

    if potential_on:
        for i in range(n_cpu):
            for j in range(len(simulations_batch[i])):
                j+= i*offset
                pool.apply_async(render_positions_w_potential, args=(simulation, j, output_dir, vmin, vmax, skip_frames))
    else:
        for i in range(n_cpu):
            for j in range(len(simulations_batch[i])):
                j+= i*offset
                pool.apply_async(render_positions, args=(simulation, j, output_dir, skip_frames))

    pool.close()
    pool.join()

    image_files = natsorted([os.path.join(output_dir, img) for img in os.listdir(output_dir) if img.endswith(".png")])

    # generate mp4 from plot pngs using moviepy
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f"animations/{file_name}.mp4")

    # delete plot pngs and remove directory
    _ = [os.remove(image_file) for image_file in image_files]
    os.rmdir(output_dir)