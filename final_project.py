#!/usr/bin/env python3

import math
import numpy as np
import astropy.units as u 
import astropy.constants as c 
import time
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation

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

    def print_total_vec(self):
        return self.total_vec
    
    def ODEs_to_solve(self, ODEs):
        self.ODEs = ODEs

    def method_to_use(self, method):
        self.method = method

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
    
    
    def leapfrog(self, t, dt):
        half_dt = 0.5 * dt

        if self.ODEs.__name__ == 'electrostatic_force':

            # Update velocities at half step
            self.total_vec[:, :3] += half_dt * self.total_vec[:, 3:]

            # Update positions
            self.total_vec[:, :3] += dt * self.total_vec[:, 3:]

            # Compute forces at new positions
            forces = self.ODEs(t + dt, self.total_vec, self.charges)

            # Update velocities at full step
            self.total_vec[:, 3:] += dt * forces[:, :3]

            # Update velocities at half step
            self.total_vec[:, 3:] += half_dt * forces[:, :3]

        elif self.ODEs.__name__ == 'gravitational_force':

            # Update velocities at half step
            self.total_vec[:, :3] += half_dt * self.total_vec[:, 3:]

            # Update positions
            self.total_vec[:, :3] += dt * self.total_vec[:, 3:]

            # Compute forces at new positions
            forces = self.ODEs(t + dt, self.total_vec, self.masses)

            # Update velocities at full step
            self.total_vec[:, 3:] += dt * forces[:, :3]

            # Update velocities at half step
            self.total_vec[:, 3:] += half_dt * forces[:, :3]

        else:
            raise Exception('Not a valid ODE')
        

    def evolve(self, t_final, dt, t_i=0):

        if self.has_units:
            t_final = t_final.cgs.value
            dt = dt.cgs.value
        else:
            t_final = t_final
            dt = dt


        self.vec_store = [self.total_vec]
        n = int((t_final-t_i)/dt)

        T = t_i
        compute_start = time.time()

        for i in range(n):
            sys.stdout.flush()
            sys.stdout.write('Integrating: step = {} / {} | simulation time = {}'.format(i,n,round(T,3)) + '\r')
            if self.method.__name__ == "RK4":
                new_vec = self.RK4(T, dt)
            elif self.method.__name__ == 'leapfrog':
                new_vec = self.leapfrog(T, dt)
            self.vec_store.append(new_vec)
            self.total_vec = new_vec
            T += dt

        compute_end = time.time()
        runtime = compute_end - compute_start
        print('\n')
        print('Simulation completed in {} seconds'.format(runtime))
        self.vec_store = np.array(self.vec_store)


def gravitational_force(t, vec, masses):

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
simulation.evolve(100*u.day,1*u.hr)

positions = simulation.vec_store[:, :, :3]

fig, ax = plt.subplots()

bodies, = ax.plot([], [], 'o')

def animate_orbits(frame):
    bodies.set_xdata(positions[frame, :, 0]) 
    bodies.set_ydata(positions[frame, :, 1])

    ax.set_xlim(np.min(positions[:, :, 0])-1e10, np.max(positions[:, :, 0])+1e10)
    ax.set_ylim(np.min(positions[:, :, 1])-1e10, np.max(positions[:, :, 1])+1e10)

    return bodies, 

animation = FuncAnimation(fig, animate_orbits, frames=positions.shape[0], interval=.5)
plt.show()