#!/usr/bin/env python3

import math
import numpy as np
import astropy.units as u 
import astropy.constants as c 
import time
import sys

def gravitational_force(t, vec, masses, G_is_one=None, softening=None):

    bodies = len(vec)  # retrieve number of bodies based on vec length

    new_vec = np.zeros_like(vec)

    for i in range(bodies):
        
        pos_i = vec[i,:3]
        vel_i = vec[i, 3:]

        for j in range(bodies):

            if i!= j:

                dx = pos_i - vec[j, :3]
                
                if softening:
                    r = (dx[0]**2 + dx[1]**2 + dx[2]**2 + softening**2)
                else:
                    r = np.linalg.norm(dx)

                # Gravitational force
                if G_is_one:
                    g = 1
                else:
                    g = c.G.si.value   # Gravitational constant

                new_vec[i, :3] = vel_i
                new_vec[i, 3] += (-g*masses[j]/ r**3) * dx[0]
                new_vec[i, 4] += (-g*masses[j]/ r**3) * dx[1]
                new_vec[i, 5] += (-g*masses[j]/ r**3) * dx[2]

    return new_vec

def electrostatic_force(t, vec, charges, has_units=False, softening=None):

    bodies = len(vec)  # retrieve number of bodies based on vec length

    new_vec = np.zeros_like(vec)

    for i in range(bodies):
        
        pos_i = vec[i,:3]
        vel_i = vec[i, 3:]

        for j in range(bodies):

            if i != j:

                dx = pos_i - vec[j, :3]
                
                if softening:
                    r = (dx[0]**2 + dx[1]**2 + dx[2]**2 + softening**2)
                else:
                    r = np.linalg.norm(dx)

                # Coulomb's law 
                k = 8.9875e9  # Coulomb's constant in N m^2 / C^2

                new_vec[i, :3] = vel_i

                new_vec[i, 3] += (k*charges[j] / r**3) * dx[0]
                new_vec[i, 4] += (k*charges[j] / r**3) * dx[1]
                new_vec[i, 5] += (k*charges[j] / r**3) * dx[2]

    return new_vec