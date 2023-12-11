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
import multiprocessing as mp

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

        self.potential_store = []

        n = int((t_final-t_i)/dt)

        T = t_i
        compute_start = time.time()

        print('Begin sim:')

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

        compute_end = time.time()
        runtime = compute_end - compute_start
        print('\n')
        print('Simulation completed in {} seconds'.format(runtime))
        self.vec_store = np.array(self.vec_store)


        # print('Generating potential grids:')
        # compute_start = time.time()

        # for i in range(self.vec_store.shape[0]):
        #     sys.stdout.flush()
        #     sys.stdout.write('Generating potential grid: step = {} / {} | simulation time = {}'.format(i+1,n,round(T,3)) + '\r')
        #     new_potential_grid = self.generate_potential_energy_grid(i)

        #     self.potential_store.append(new_potential_grid.copy())

        # compute_end = time.time()
        # runtime = compute_end - compute_start
        # print('\n')
        # print('Potential Grid Computation completed in {} seconds'.format(runtime))
        # self.potential_store = np.array(self.potential_store)
    
    ####### COMPUTE ENERGIES #################

    def generate_potential_energy_grid(self, dt, resolution=100):

        xmin, xmax = np.min(self.vec_store[:, :, 0])-1e12, np.max(self.vec_store[:, :, 0])+1e12
        ymin, ymax = np.min(self.vec_store[:, :, 1])-1e12, np.max(self.vec_store[:, :, 1])+1e12
        
        x_range = np.linspace(xmin, xmax, resolution)
        y_range = np.linspace(ymin, ymax, resolution)

        X, Y = np.meshgrid(x_range, y_range)

        potential_grid = np.zeros_like(X)

        for i in range(len(x_range)):
            for j in range(len(y_range)):    
                position = np.array([X[i, j], Y[i, j], 0])
                for n in range(self.N):
                    potential_grid[i, j] += compute_grav_potential_energy(self.vec_store[dt, n, :3], self.masses[n], position)
    
        return np.log10(potential_grid)

def gravitational_force(t, vec, masses, softening=None):

    bodies = len(vec)  # retrieve number of bodies based on vec length

    new_vec = np.zeros_like(vec)

    for i in range(bodies):
        
        pos_i = vec[i,:3]
        vel_i = vec[i, 3:]

        for j in range(bodies):

            if i!= j:

                dx = pos_i - vec[j, :3]
                
                r = np.linalg.norm(dx)

                # Gravitational force
                g = 1  # Gravitational constant

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
                
                r = np.linalg.norm(dx)

                # Coulomb's law 
                k = 8.9875e9  # Coulomb's constant in N m^2 / C^2

                new_vec[i, :3] = vel_i

                new_vec[i, 3] += (k*charges[j] / r**3) * dx[0]
                new_vec[i, 4] += (k*charges[j] / r**3) * dx[1]
                new_vec[i, 5] += (k*charges[j] / r**3) * dx[2]

    return new_vec


def compute_grav_potential_energy(body_position, mass, position):
        
        g = 1  # Gravitational constant
        
        dx = position - body_position

        r = np.linalg.norm(dx)

        U = g * mass / r

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

    ax.set_title(f'Orbits at {dt}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
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
    fig.colorbar(pcm, ax=ax, label='Gravitational Potential')

    plt.savefig(os.path.join(output_dir, f'orbit_timestep_{dt}.png'))
    plt.close(fig)