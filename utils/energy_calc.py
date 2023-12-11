#!/usr/bin/env python3

import math
import numpy as np
import astropy.units as u 
import astropy.constants as c 
import time
import sys

def calculate_total_energy(vec, masses, G_is_one=False):

    num_bodies = len(masses)

    for i in range(num_bodies):
        pos_i = vec[i, :3]
        vel_i = vec[i, 3:]

        KE = 0.0
        U = 0.0

        KE += 0.5 * masses[i] * np.linalg.norm(vel_i)**2

        for j in range(num_bodies):
            if i != j:
                # Calculate the distance between the bodies
                r = np.linalg.norm(pos_i - vec[j, :3])

                # Calculate the gravitational potential energy
                if G_is_one:
                    G = 1.0  # Gravitational constant
                else:
                    G = c.G.si.value  # Actual gravitational constant

                U += - G * masses[i] * masses[j] / r

    total_energy = 0
    total_energy += KE + U

    return total_energy, KE, U

def compute_grav_potential_energy(body_position, mass, position, G_is_one=False, softening=1e-3):
        
        if G_is_one:
            g = 1
        else:
            g = c.G.si.value   # Gravitational constant
        
        dx = position - body_position

        r = np.linalg.norm(dx)

        if softening:
            r = np.sqrt(np.sum(dx**2) + softening**2)

        U = g * mass / r

        return U