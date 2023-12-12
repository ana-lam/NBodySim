#!/usr/bin/env python3

import numpy as np
import astropy.constants as c 


"""Compute gravitational force
t: current time step
vec: pos and vel
dimensionless: set G=1 or G=6.6743e-11
softening: softening parameter to avoid blow up when r->0
"""
def gravitational_force(t, vec, masses, dimensionless=False, softening=1e-3):

    bodies = len(vec) 
    new_vec = np.zeros_like(vec)

    for i in range(bodies):
        
        pos_i = vec[i,:3]
        vel_i = vec[i, 3:]

        for j in range(bodies):

            if i!= j:
            
                # calculate the distance between the bodies
                dx = pos_i - vec[j, :3]
                r = np.linalg.norm(dx)

                if softening:
                    r = np.sqrt(r**2 + softening**2)

                if dimensionless:
                    G = 1
                else:
                    G = c.G.si.value        # Gravitational constant in m^3 / kg s^2

                # solve law of gravitation
                new_vec[i, :3] = vel_i
                new_vec[i, 3] += (-G*masses[j]/ r**3) * dx[0]
                new_vec[i, 4] += (-G*masses[j]/ r**3) * dx[1]
                new_vec[i, 5] += (-G*masses[j]/ r**3) * dx[2]

    return new_vec


"""Compute electrostatic force
t: current time step
vec: pos and vel
dimensionless: set k=1 or k=8.9875e9
softening: softening parameter to avoid blow up when r->0
"""
def electrostatic_force(t, vec, charges, dimensionless=False, softening=1e-3):

    bodies = len(vec)
    new_vec = np.zeros_like(vec)

    for i in range(bodies):
        
        pos_i = vec[i,:3]
        vel_i = vec[i, 3:]

        for j in range(bodies):

            if i != j:

                # calculate the distance between the bodies
                dx = pos_i - vec[j, :3]
                r = np.linalg.norm(dx)

                if softening:
                    r = np.sqrt(r**2 + softening**2)

                if dimensionless:
                    k=1
                else:
                    k = 8.9875e9        # Coulomb's constant in N m^2 / C^2

                # solve Coulomb's law
                new_vec[i, :3] = vel_i
                new_vec[i, 3] += (k*charges[j] / r**3) * dx[0]
                new_vec[i, 4] += (k*charges[j] / r**3) * dx[1]
                new_vec[i, 5] += (k*charges[j] / r**3) * dx[2]

    return new_vec