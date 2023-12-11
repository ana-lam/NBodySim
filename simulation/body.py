#!/usr/bin/env python3

import numpy as np

class Body():
    
    def __init__(self, mass, x_vec, v_vec, charge=None, has_units=True):
        
        self.has_units = has_units
        self.charge = charge
        self.mass = mass
        self.x_vec = x_vec
        self.v_vec = v_vec
        
    def vec(self):
        # Return position (x, y, z) and velocity (vx, vy, vz) components in array
        return np.concatenate( (self.x_vec, self.v_vec) )