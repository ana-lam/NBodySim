#!/usr/bin/env python3

import math
import numpy as np
import astropy.units as u 
import astropy.constants as c 
import time
import sys
from simulation.forces import gravitational_force, electrostatic_force
from utils.energy_calc import compute_grav_potential_energy, calculate_total_energy

class System():
    
    def __init__(self, bodies, has_units=True, G_is_one=True):
        self.has_units = has_units
        self.bodies = bodies
        self.N = len(bodies)
        self.masses = np.array([i.mass for i in self.bodies])
        self.total_vec = np.array([i.vec() for i in self.bodies])
        self.G_is_one = G_is_one
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

            k1 = dt * self.ODEs(t, self.total_vec, self.charges, self.G_is_one)
            k2 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k1, self.charges, self.G_is_one)
            k3 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k2, self.charges, self.G_is_one)
            k4 = dt * self.ODEs(t+dt, self.total_vec+k3, self.charges, self.G_is_one)

            new_vec = self.total_vec + ((k1+ 2*k2 + 2*k3 + k4)/6)

        elif self.ODEs.__name__ == 'gravitational_force':
            
            k1 = dt * self.ODEs(t, self.total_vec, self.masses, self.G_is_one)
            k2 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k1, self.masses, self.G_is_one)
            k3 = dt * self.ODEs(t+0.5*dt, self.total_vec+0.5*k2, self.masses, self.G_is_one)
            k4 = dt * self.ODEs(t+dt, self.total_vec+k3, self.masses, self.G_is_one)

            new_vec = self.total_vec + ((k1+ 2*k2 + 2*k3 + k4)/6)

        else:
            raise Exception('Not a valid ODE')
        
        return new_vec

    ##### LEAPFROG ############################
    
    def leapfrog(self, t, dt):

        if self.ODEs.__name__ == 'electrostatic_force':
            
            self.total_vec[:, 3:] += 0.5 * dt * self.ODEs(t, self.total_vec, self.charges, self.G_is_one)[:, 3:]

            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += 0.5*dt * self.ODEs(t, self.total_vec, self.charges, self.G_is_one)[:, 3:]

            new_vec = self.total_vec

        elif self.ODEs.__name__ == 'gravitational_force':


            self.total_vec[:, 3:] += 0.5 * dt * self.ODEs(t, self.total_vec, self.masses, self.G_is_one)[:, 3:]

            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += 0.5*dt * self.ODEs(t, self.total_vec, self.masses, self.G_is_one)[:, 3:]

            new_vec = self.total_vec

        else:
            raise Exception('Not a valid ODE')

        return new_vec
    
    ##### VELOCITY VERLET #####################

    def velocityVerlet(self, t, dt):

        if self.ODEs.__name__ == 'electrostatic_force': 

            old_acc = self.ODEs(t, self.total_vec, self.charges, self.G_is_one)[:, 3:]

            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt + .5 * old_acc * dt**2

            new_acc = self.ODEs(t, self.total_vec, self.charges, self.G_is_one)[:, 3:]

            self.total_vec[:, 3:] += .5 * (old_acc + new_acc)*dt

            new_vec = self.total_vec
            

        elif self.ODEs.__name__ == 'gravitational_force':
            
            old_acc = self.ODEs(t, self.total_vec, self.masses, self.G_is_one)[:, 3:]

            self.total_vec[:, :3] += self.total_vec[:, 3:] * dt + .5 * old_acc * dt**2

            new_acc = self.ODEs(t, self.total_vec, self.masses, self.G_is_one)[:, 3:]

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
            
            self.total_vec[:, 3:] += d1*self.ODEs(t, self.total_vec, self.charges, self.G_is_one)[:, 3:] * dt

            self.total_vec[:, :3] += c2*self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += d2*self.ODEs(t, self.total_vec, self.charges, self.G_is_one)[:, 3:] * dt

            self.total_vec[:, :3] += c3*self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += d3*self.ODEs(t, self.total_vec, self.charges, self.G_is_one)[:, 3:] * dt

            new_vec = self.total_vec
        
        elif self.ODEs.__name__ == 'gravitational_force':

            self.total_vec[:, :3] += c1 * self.total_vec[:, 3:] * dt
            
            self.total_vec[:, 3:] += d1*self.ODEs(t, self.total_vec, self.masses, self.G_is_one)[:, 3:] * dt

            self.total_vec[:, :3] += c2*self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += d2*self.ODEs(t, self.total_vec, self.masses, self.G_is_one)[:, 3:] * dt

            self.total_vec[:, :3] += c3*self.total_vec[:, 3:] * dt

            self.total_vec[:, 3:] += d3*self.ODEs(t, self.total_vec, self.masses, self.G_is_one)[:, 3:] * dt

            self.total_vec[:, :3] += c4*self.total_vec[:, 3:] * dt

            new_vec = self.total_vec

        return new_vec
        
    ######### TIME STEPPING ###############

    def evolve(self, t_final, dt, t_i=0):

        if self.has_units:
            t_final = t_final.si.value
            dt = dt.si.value
        else:
            t_final = t_final
            dt = dt

        # self.potential_energy_store = []

        self.vec_store = [self.total_vec]

        ## energy stores
        self.energy_store = []
        self.KE_store = []
        self.U_store = []

        n = int((t_final-t_i)/dt)

        self.time_steps = np.linspace(t_i, t_final, n)

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

        print('Begin energy calc:')
        compute_start = time.time()
        for i in range(n):
            sys.stdout.flush()
            sys.stdout.write('Calculating: step = {} / {} | energy calc time = {}'.format(i+1,n,round(T,3)) + '\r')
            total_energy, KE, U = calculate_total_energy(self.vec_store[i], self.masses, self.G_is_one)
            self.energy_store.append(total_energy.copy())
            self.KE_store.append(KE.copy())
            self.U_store.append(U.copy())
        compute_end = time.time()
        runtime = compute_end - compute_start
        print('\n')
        print('Energy calc completed in {} seconds'.format(runtime))
        self.energy_store = np.array(self.energy_store)
        self.KE_store = np.array(self.KE_store)
        self.U_store = np.array(self.U_store)


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
                    potential_grid[i, j] += compute_grav_potential_energy(self.vec_store[dt, n, :3], self.masses[n], position, G_is_one=self.G_is_one)
    
        return np.log10(potential_grid)