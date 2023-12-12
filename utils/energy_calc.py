#!/usr/bin/env python3

import numpy as np 
import astropy.constants as c 


"""Calculate total energy, kinetic energy, & potential energy.
vec: pos and vel
masses: array of N masses
dimensionless: set G=1 or G=6.6743e-11, k=1 or k=8.9875e9
"""
def calculate_energies(vec, masses, dimensionless=False):

    num_bodies = len(masses)

    for i in range(num_bodies):
        pos_i = vec[i, :3]
        vel_i = vec[i, 3:]

        KE = 0.0
        U = 0.0

        # calculate kinetic energy
        KE += 0.5 * masses[i] * np.linalg.norm(vel_i)**2

        for j in range(num_bodies):
            if i != j:
                # calculate the distance between positions
                r = np.linalg.norm(pos_i - vec[j, :3])

                if dimensionless:
                    G = 1.0  
                else:
                    G = c.G.si.value        # Gravitational constant in m^3 / kg s^2

                # calculate potential energy
                U += - G * masses[i] * masses[j] / r

    total_energy = 0
    total_energy += KE + U

    return total_energy, KE, U


"""Calculate total energy, kinetic energy, & potential energy.
body_position: pos of body i
mass: mass of body i
dimensionless: set G=1 or G=6.6743e-11, k=1 or k=8.9875e9
softening: softening parameter to avoid blow up when r->0
"""
def compute_grav_potential_energy(body_position, mass, position, dimensionless=False, softening=1e-3):
        
        if dimensionless:
            G = 1
        else:
            G = c.G.si.value   # Gravitational constant
        
        # calculate the distance between positions
        dx = position - body_position
        r = np.linalg.norm(dx)

        if softening:
            r = np.sqrt(np.sum(dx**2) + softening**2)

        # calculate potential energy
        U = G * mass / r

        return U