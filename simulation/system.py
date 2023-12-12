#!/usr/bin/env python3

import numpy as np
import astropy.units as u 
import time
import sys
from utils.energy_calc import compute_grav_potential_energy, calculate_energies


""" System Instance
__init__: initialize System with parameters
ODEs_to_solve: sets ODE to solve
method_to_use: sets integration method to use
RK4: Runge-Kutta 4th Order to update pos-vel vector
leapfrog: Leapfrog to update pos-vel vector
velocity_verlet: Velocity-Verlet to update pos-vel vector
yoshida: Yoshida 4th Order to update pos-vel vector
evolve: Method that begins system time evolution
generate_potential_energy_grid: generates potential energy grid
"""
class System():
   
    
    """Initialize System object instance from Body objects
    """
    def __init__(self, bodies, dimensionless=True):
        
        self.bodies = bodies                                            # array of Body instances (len N)
        self.N = len(bodies)                                            # number of bodies in system
        self.masses = np.array([i.mass for i in self.bodies])           # array of all masses (len N)
        self.total_vec = np.array([i.vec() for i in self.bodies])       # N-D array of all x, y, z, vx, vy, vz (len N, len 6)
        self.dimensionless = dimensionless                              # constant dimensionless/not dimensionless
        if bodies[0].charge:
            self.charges = np.array([i.charge for i in self.bodies])    # array of all charges (len N)


    """Set ODE to solve from forces.py
    - gravitational_force:  F=G*m_i*m_j/r**2
    - electrostatic_force:  F=k*q_i*q_j/r**2
    """
    def ODEs_to_solve(self, ODEs):                                     
        self.ODEs = ODEs


    """Set integration method to use.
    - Runge-Kutta 4th Order
    - Leapfrog
    - Velocity-Verlet
    - Yoshida 4th Order
    """
    def method_to_use(self, method):                                    
        self.method = method


    """RUNGE_KUTTA 4TH ORDER CALCULATION.
    t: current time step
    dt: time step size
    """
    def RK4(self, t, dt):

        if self.ODEs.__name__ == 'electrostatic_force':

            k1 = dt * self.ODEs(t, self.total_vec, self.charges, self.dimensionless)
            k2 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k1, self.charges, self.dimensionless)
            k3 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k2, self.charges, self.dimensionless)
            k4 = dt * self.ODEs(t+dt, self.total_vec+k3, self.charges, self.dimensionless)

            new_vec = self.total_vec + ((k1+ 2*k2 + 2*k3 + k4)/6)

        elif self.ODEs.__name__ == 'gravitational_force':
            
            k1 = dt * self.ODEs(t, self.total_vec, self.masses, self.dimensionless)
            k2 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k1, self.masses, self.dimensionless)
            k3 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k2, self.masses, self.dimensionless)
            k4 = dt * self.ODEs(t+dt, self.total_vec+k3, self.masses, self.dimensionless)

            new_vec = self.total_vec + ((k1+ 2*k2 + 2*k3 + k4)/6)

        else:
            raise Exception('Not a valid ODE')
        
        return new_vec


    """LEAPFROG CALCULATION.
    t: current time step
    dt: time step size
    """
    def leapfrog(self, t, dt):

        if self.ODEs.__name__ == 'electrostatic_force':
            
            self.total_vec[:, 3:] += 0.5 * dt * self.ODEs(t, self.total_vec, self.charges, self.dimensionless)[:, 3:]
            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt
            self.total_vec[:, 3:] += 0.5*dt * self.ODEs(t, self.total_vec, self.charges, self.dimensionless)[:, 3:]

            new_vec = self.total_vec

        elif self.ODEs.__name__ == 'gravitational_force':

            self.total_vec[:, 3:] += 0.5 * dt * self.ODEs(t, self.total_vec, self.masses, self.dimensionless)[:, 3:]
            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt
            self.total_vec[:, 3:] += 0.5*dt * self.ODEs(t, self.total_vec, self.masses, self.dimensionless)[:, 3:]

            new_vec = self.total_vec

        else:
            raise Exception('Not a valid ODE')

        return new_vec
    

    """VELOCITY-VERLET CALCULATION.
    t: current time step
    dt: time step size
    """
    def velocity_verlet(self, t, dt):

        if self.ODEs.__name__ == 'electrostatic_force': 

            old_acc = self.ODEs(t, self.total_vec, self.charges, self.dimensionless)[:, 3:]
            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt + .5 * old_acc * dt**2
            new_acc = self.ODEs(t, self.total_vec, self.charges, self.dimensionless)[:, 3:]
            self.total_vec[:, 3:] += .5 * (old_acc + new_acc)*dt

            new_vec = self.total_vec
            
        elif self.ODEs.__name__ == 'gravitational_force':
            
            old_acc = self.ODEs(t, self.total_vec, self.masses, self.dimensionless)[:, 3:]
            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt + .5 * old_acc * dt**2
            new_acc = self.ODEs(t, self.total_vec, self.masses, self.dimensionless)[:, 3:]
            self.total_vec[:, 3:] += .5 * (old_acc + new_acc)*dt

            new_vec = self.total_vec

        return new_vec
    

    """YOSHIDA 4TH ORDER CALCULATION.
    t: current time step
    dt: time step size
    """
    def yoshida(self, t, dt):

        # define Yoshida coefficients
        b = 2 ** (1/3)

        w1 = 1 / (2-b)
        w0 = -b * w1

        c1 = c4 = w1 * .5
        c2 = c3 = (w0+w1) * .5
        d1 = d3 = w1
        d2 = w0

        if self.ODEs.__name__ == 'electrostatic_force': 

            self.total_vec[:, :3] += c1 * self.total_vec[:, 3:] * dt
            self.total_vec[:, 3:] += d1*self.ODEs(t, self.total_vec, self.charges, self.dimensionless)[:, 3:] * dt
            self.total_vec[:, :3] += c2*self.total_vec[:, 3:] * dt
            self.total_vec[:, 3:] += d2*self.ODEs(t, self.total_vec, self.charges, self.dimensionless)[:, 3:] * dt
            self.total_vec[:, :3] += c3*self.total_vec[:, 3:] * dt
            self.total_vec[:, 3:] += d3*self.ODEs(t, self.total_vec, self.charges, self.dimensionless)[:, 3:] * dt

            new_vec = self.total_vec
        
        elif self.ODEs.__name__ == 'gravitational_force':

            self.total_vec[:, :3] += c1 * self.total_vec[:, 3:] * dt   
            self.total_vec[:, 3:] += d1*self.ODEs(t, self.total_vec, self.masses, self.dimensionless)[:, 3:] * dt
            self.total_vec[:, :3] += c2*self.total_vec[:, 3:] * dt
            self.total_vec[:, 3:] += d2*self.ODEs(t, self.total_vec, self.masses, self.dimensionless)[:, 3:] * dt
            self.total_vec[:, :3] += c3*self.total_vec[:, 3:] * dt
            self.total_vec[:, 3:] += d3*self.ODEs(t, self.total_vec, self.masses, self.dimensionless)[:, 3:] * dt
            self.total_vec[:, :3] += c4*self.total_vec[:, 3:] * dt

            new_vec = self.total_vec

        return new_vec
        
    """Evolve the N body system.
    t_final: end time of time interval
    dt: time step size
    t_i: initial time of time interval
    """
    def evolve(self, t_final, dt, t_i=0):

        if isinstance(t_final, u.Quantity):
            t_final = t_final.si.value

        if isinstance(dt, u.Quantity):
            dt = dt.si.value

        self.vec_store = [self.total_vec]               # store for pos vel vector at each time step

        self.energy_store = []                          # stores for total energy at each time step
        self.KE_store = []                              # stores for kinetic energy at each time step
        self.U_store = []                               # stores for potential energy at each time step

        n = int((t_final-t_i)/dt)                       # number of time steps
        self.time_steps = np.linspace(t_i, t_final, n)

        compute_start = time.time()

        print('Begin sim:')

        if self.method == "RK4":
            print("Using RK4...")
        elif self.method == 'leapfrog':
            print("Using leapfrog...")
        elif self.method == 'velocityVerlet':
            print("Using velocityVerlet...")
        elif self.method == 'yoshida':
            print("Using yoshida...")

        # evolve the system and update pos vel vector + energy
        T = t_i
        for i in range(n):
            sys.stdout.flush()
            sys.stdout.write('Integrating: step = {} / {} '.format(i+1,n) + '\r')
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

            total_energy, KE, U = calculate_energies(self.vec_store[i], self.masses, self.G_is_one)
            self.energy_store.append(total_energy.copy())
            self.KE_store.append(KE.copy())
            self.U_store.append(U.copy())

        compute_end = time.time()
        runtime = compute_end - compute_start
        print('Simulation completed in {} seconds'.format(runtime))
        
        self.vec_store = np.array(self.vec_store)
        self.energy_store = np.array(self.energy_store)
        self.KE_store = np.array(self.KE_store)
        self.U_store = np.array(self.U_store)

        # print('\n')
        # print('Begin energy calc:')
        # compute_start = time.time()

        # # calculate energy for each time step
        # for i in range(n):
        #     sys.stdout.flush()
        #     sys.stdout.write('Calculating: step = {} / {}'.format(i+1,n) + '\r')
        #     total_energy, KE, U = calculate_energies(self.vec_store[i], self.masses, self.G_is_one)
        #     self.energy_store.append(total_energy.copy())
        #     self.KE_store.append(KE.copy())
        #     self.U_store.append(U.copy())
        # compute_end = time.time()
        # runtime = compute_end - compute_start

        # print('Energy calc completed in {} seconds'.format(runtime))
        # self.energy_store = np.array(self.energy_store)
        # self.KE_store = np.array(self.KE_store)
        # self.U_store = np.array(self.U_store)


    """Generate potential energy grid.
    dt: time step size
    resolution: resolution of grid, number of grid spaces
    """
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
                    potential_grid[i, j] += compute_grav_potential_energy(self.vec_store[dt, n, :3], 
                    self.masses[n], position, dimensionless=self.dimensionless)
    
        return np.log10(potential_grid)