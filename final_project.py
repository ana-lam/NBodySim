#!/usr/bin/env python3

import math
import numpy as np
import astropy.units as u 
import astropy.constants as c 
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from IPython.display import HTML
import os
from natsort import natsorted
import moviepy.video.io.ImageSequenceClip
import multiprocessing.pool
from plotter import render_positions, plot_wrapper

class Body():
    
    def __init__(self, mass, x_vec, v_vec, charge=None, has_units=True):
        
        self.has_units = has_units
        self.charge = charge
        
        # take value only if units
        if self.has_units:
            self.mass = mass.cgs.value
            self.x_vec = x_vec.cgs.value
            self.v_vec = v_vec.cgs.value
            if self.charge:
                self.charge = charge.value
        
        else:
            self.mass = mass
            self.x_vec = x_vec
            self.v_vec = v_vec
            if self.charge:
                self.charge = charge
        
    def vec(self):
        return np.concatenate( (self.x_vec, self.v_vec) ) # store all pos and vel components for body

class System():
    
    def __init__(self, bodies, has_units=True):
        self.has_units = has_units
        self.bodies = bodies
        self.N = len(bodies)
        self.masses = np.array([i.mass for i in self.bodies])
        self.total_vec = np.array([i.vec() for i in self.bodies])
        if bodies[0].charge:
            self.charges = np.array([i.charge for i in self.bodies])

    def return_masses(self):
        return self.masses
    
    def print_total_vec(self):
        return self.total_vec
    
    def ODEs_to_solve(self, ODEs):
        self.ODEs = ODEs

    def method_to_use(self, method):
        self.method = method

    #### RUNGE KUTTA 4th ORDER #################

    def RK4(self, t, dt):

        if self.ODEs.__name__ == 'electrostatic_force':

            k1 = dt * self.ODEs(t, self.total_vec, self.charges)
            k2 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k1, self.charges)
            k3 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k2, self.charges)
            k4 = dt * self.ODEs(t+dt, self.total_vec+k3, self.charges)

            new_vec = self.total_vec + ((k1+ 2*k2 + 2*k3 + k4)/6)

        elif self.ODEs.__name__ == 'gravitational_force':
            
            k1 = dt * self.ODEs(t, self.total_vec, self.masses)
            k2 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k1, self.masses)
            k3 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k2, self.masses)
            k4 = dt * self.ODEs(t+dt, self.total_vec+k3, self.masses)

            new_vec = self.total_vec + ((k1+ 2*k2 + 2*k3 + k4)/6)

        else:
            raise Exception('Not a valid ODE')
        
        return new_vec

    ##### LEAPFROG ############################
    
    def leapfrog(self, t, dt):

        if self.ODEs.__name__ == 'electrostatic_force':
            
            self.total_vec[:, 3:] += 0.5 * dt * self.ODEs(t, self.total_vec, self.charges)[:, 3:]

            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += 0.5*dt * self.ODEs(t, self.total_vec, self.charges)[:, 3:]

            new_vec = self.total_vec

        elif self.ODEs.__name__ == 'gravitational_force':


            self.total_vec[:, 3:] += 0.5 * dt * self.ODEs(t, self.total_vec, self.masses)[:, 3:]

            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += 0.5*dt * self.ODEs(t, self.total_vec, self.masses)[:, 3:]

            new_vec = self.total_vec

        else:
            raise Exception('Not a valid ODE')

        return new_vec
    
    ##### VELOCITY VERLET #####################

    def velocityVerlet(self, t, dt):

        if self.ODEs.__name__ == 'electrostatic_force': 

            old_acc = self.ODEs(t, self.total_vec, self.charges)[:, 3:]

            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt + .5 * old_acc * dt**2

            new_acc = self.ODEs(t, self.total_vec, self.charges)[:, 3:]

            self.total_vec[:, 3:] += .5 * (old_acc + new_acc)*dt

            new_vec = self.total_vec
            

        elif self.ODEs.__name__ == 'gravitational_force':
            
            old_acc = self.ODEs(t, self.total_vec, self.masses)[:, 3:]

            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt + .5 * old_acc * dt**2

            new_acc = self.ODEs(t, self.total_vec, self.masses)[:, 3:]

            self.total_vec[:, 3:] += .5 * (old_acc + new_acc)*dt

            new_vec = self.total_vec

        return new_vec
    

    ######## YOSHIDA 4TH ORDER ################

    def yoshida(self, t, dt):

        # first calculate Yoshida coefficients c1, c2, c3, c4 and d1, d2, d3

        b = 2 ** (1/3)  # cube root of 2

        w1 = 1 / (2-b)
        w0 = -b * w1

        c1 = c4 = w1 * .5
        c2 = c3 = (w0+w1) * .5
        
        d1 = d3 = w1
        d2 = w0

        if self.ODEs.__name__ == 'electrostatic_force': 

            self.total_vec[:, :3] += c1 * self.total_vec[:, 3:] * dt
            
            self.total_vec[:, 3:] += d1*self.ODEs(t, self.total_vec, self.charges)[:, 3:] * dt

            self.total_vec[:, :3] += c2*self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += d2*self.ODEs(t, self.total_vec, self.charges)[:, 3:] * dt

            self.total_vec[:, :3] += c3*self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += d3*self.ODEs(t, self.total_vec, self.charges)[:, 3:] * dt

            new_vec = self.total_vec
        
        elif self.ODEs.__name__ == 'gravitational_force':

            self.total_vec[:, :3] += c1 * self.total_vec[:, 3:] * dt
            
            self.total_vec[:, 3:] += d1*self.ODEs(t, self.total_vec, self.masses)[:, 3:] * dt

            self.total_vec[:, :3] += c2*self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += d2*self.ODEs(t, self.total_vec, self.masses)[:, 3:] * dt

            self.total_vec[:, :3] += c3*self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += d3*self.ODEs(t, self.total_vec, self.masses)[:, 3:] * dt

            new_vec = self.total_vec

        return new_vec
    
    ####### COMPUTE ENERGIES #################

    def compute_potential_energy(self, position):
        
        U=0
        for i in range(self.N):
            for j in range(self.N):
                if i != j:

                    dx = position - self.total_vec[i, :3]
                    r = (dx[0]**2 + dx[1]**2 + dx[2]**2)**0.5

                    g = 6.674e-11  # Gravitational constant

                    U += -g * self.masses[i]*self.masses[j] / r

        return U
        
    ######### TIME STEPPING ###############

    def evolve(self, t_final, dt, t_i=0):

        if self.has_units:
            t_final = t_final.cgs.value
            dt = dt.cgs.value
        else:
            t_final = t_final
            dt = dt

        # self.potential_energy_store = []

        self.vec_store = [self.total_vec]
        n = int((t_final-t_i)/dt)

        T = t_i
        compute_start = time.time()

        for i in range(n):
            sys.stdout.flush()
            sys.stdout.write('Integrating: step = {} / {} | simulation time = {}'.format(i+1,n,round(T,3)) + '\r')
            if self.method == "RK4":
                new_vec = self.RK4(T, dt)
            elif self.method == 'leapfrog':
                new_vec = self.leapfrog(T, dt)
            elif self.method == 'velocityVerlet':
                new_vec = self.velocityVerlet(T, dt)
            elif self.method == 'yoshida':
                new_vec = self.yoshida(T, dt)
            
            # this boy caused a lot of problems so I had to add .copy() lol
            self.vec_store.append(new_vec.copy())
            
            self.total_vec = new_vec
            T += dt

            # potential_energy = self.compute_potential_energy()
            # self.potential_energy_store.append(potential_energy)

        compute_end = time.time()
        runtime = compute_end - compute_start
        print('\n')
        print('Simulation completed in {} seconds'.format(runtime))
        self.vec_store = np.array(self.vec_store)

        # self.potential_energy_store = np.array(self.potential_energy_store)


def gravitational_force(t, vec, masses, softening=None):

    bodies = len(vec)  # retrieve number of bodies based on vec length

    new_vec = np.zeros_like(vec)

    for i in range(bodies):
        
        pos_i = vec[i,:3]
        vel_i = vec[i, 3:]

        for j in range(bodies):

            if i!= j:

                dx = pos_i - vec[j, :3]
                
                r = (dx[0]**2 + dx[1]**2 + dx[2]**2)**0.5

                # Gravitational force
                g = 6.674e-11  # Gravitational constant

                new_vec[i, :3] = vel_i

                new_vec[i, 3] += (-g*masses[j]/ r**3) * dx[0]
                new_vec[i, 4] += (-g*masses[j]/ r**3) * dx[1]
                new_vec[i, 5] += (-g*masses[j]/ r**3) * dx[2]

    return new_vec

def electrostatic_force(t, vec, charges):

    bodies = len(vec)  # retrieve number of bodies based on vec length

    new_vec = np.zeros_like(vec)

    for i in range(bodies):
        
        pos_i = vec[i,:3]
        vel_i = vec[i, 3:]

        for j in range(bodies):

            if i != j:

                dx = pos_i - vec[j, :3]
                
                r = (dx[0]**2 + dx[1]**2 + dx[2]**2)**0.5

                # Coulomb's law 
                k = 8.9875e9  # Coulomb's constant in N m^2 / C^2

                new_vec[i, :3] = vel_i

                new_vec[i, 3] += (k*charges[j] / r**3) * dx[0]
                new_vec[i, 4] += (k*charges[j] / r**3) * dx[1]
                new_vec[i, 5] += (k*charges[j] / r**3) * dx[2]

    return new_vec


def compute_grav_potential_energy(simulation, dt, position):
        
        U=0
        for i in range(simulation.vec_store.shape[1]):
            for j in range(simulation.vec_store.shape[1]):
                if i != j:

                    dx = position - simulation.vec_store[dt, i, 0]
                    r = (dx[0]**2 + dx[1]**2 + dx[2]**2)**0.5

                    g = 6.674e-11  # Gravitational constant

                    U += g * simulation.masses[i]*simulation.masses[j] / r

        return U


def render_positions(simulation, dt, output_dir):
    fig, ax = plt.subplots()

    for i in range(simulation.vec_store.shape[1]):
        ax.plot(simulation.vec_store[dt, i, 0], simulation.vec_store[dt, i, 1], 'o', label=f'Body {i + 1}')

    ax.set_title(f'Orbits at {dt}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    ax.set_xlim(np.min(simulation.vec_store[:, :, 0])-1e10, np.max(simulation.vec_store[:, :, 0])+1e10)
    ax.set_ylim(np.min(simulation.vec_store[:, :, 1])-1e10, np.max(simulation.vec_store[:, :, 1])+1e10)

    plt.savefig(os.path.join(output_dir, f'orbit_timestep_{dt}.png'))
    plt.close()

def render_positions_w_potential(simulation, dt, output_dir):
    fig, ax = plt.subplots()

    xmin, xmax = np.min(simulation.vec_store[:, :, 0])-1e10, np.max(simulation.vec_store[:, :, 0])+1e10
    ymin, ymax = np.min(simulation.vec_store[:, :, 1])-1e10, np.max(simulation.vec_store[:, :, 1])+1e10

    for i in range(simulation.vec_store.shape[1]):
        ax.plot(simulation.vec_store[dt, i, 0], simulation.vec_store[dt, i, 1], 'o', label=f'Body {i + 1}')

    ax.set_title(f'Orbits at {dt}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # potential gradient

    x_range = np.linspace(xmin, xmax)
    y_range = np.linspace(ymin, ymax)

    X, Y = np.meshgrid(x_range, y_range)

    potential_grid = np.zeros_like(X)

    for i in range(len(x_range)):
        for j in range(len(y_range)):
            position = np.array([X[i, j], Y[i, j], 0])
            potential_grid[i, j] = np.log(compute_grav_potential_energy(simulation, dt, position))
            
    pcm = ax.pcolormesh(X, Y, potential_grid, cmap='viridis', shading='auto')
    fig.colorbar(pcm, ax=ax, label='Gravitational Potential')

    plt.savefig(os.path.join(output_dir, f'orbit_timestep_{dt}.png'))
    plt.close()


def create_animation(simulation, output_dir):

    dts = simulation.vec_store.shape[0]

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    n_cpu = multiprocessing.cpu_count()

    with multiprocessing.pool.ThreadPool(processes=n_cpu) as pool:
        plot_arg_list = [(simulation, dt, output_dir) for dt in range(dts)]
        result = [pool.apply_async(render_positions_w_potential, arg) for arg in plot_arg_list]
        for r in result:
            r.get()
        pool.close()
        pool.join()

    fps = 30

    image_files = natsorted([os.path.join(output_dir, img) for img in os.listdir(output_dir) if img.endswith(".png")])

    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile("animation.mp4")

    _ = [os.remove(image_file) for image_file in image_files]
    os.rmdir(output_dir)

def plot_orbits(simulation):

    fig, ax = plt.subplots()

    positions = simulation.vec_store[:, :, :3]

    for i in range(positions.shape[1]):
        ax.plot(positions[:, i, 0], positions[:, i, 1], 'o', label=f'Body {i + 1}')

    ax.set_title('Orbits')
    ax.set_xlabel('X Position')  
    ax.set_ylabel('Y Position')
    ax.legend()

    plt.show()


def plot_orbits2(simulation):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    positions = simulation.vec_store[:, :, :3]
    time_steps = np.arange(len(simulation.potential_energy_store))

    for i in range(positions.shape[1]):
        ax1.plot(positions[0, i, 0], positions[0, i, 1], 'o', label=f'Body {i + 1}')

    ax1.set_title('Orbits')
    ax1.set_ylabel('Y Position')
    ax1.legend()

    potential_energy_store = simulation.potential_energy_store
    ax2.plot(time_steps, potential_energy_store, label='Gravitational Potential Energy')
    ax2.set_title('Gravitational Potential Energy')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Energy')
    ax2.set_xlim(min(time_steps), max(time_steps))

    ax2.legend()

Body1 = Body(
    mass=2e30,
    x_vec=np.array([0, 0, 0]),
    v_vec=np.array([0, 0, 0]),
    has_units=False
)

Body2 = Body(
    mass=3.285e23,
    x_vec=np.array([0,5.7e10,0]),
    v_vec=np.array([47000,0,0]),
    has_units=False
)

bodies = [Body1, Body2]
simulation = System(bodies)
simulation.ODEs_to_solve(gravitational_force)
simulation.method_to_use("leapfrog")
simulation.evolve(365 * u.day,1*u.day)
create_animation(simulation, 'orbit_animation_output')