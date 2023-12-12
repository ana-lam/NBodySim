#!/usr/bin/env python3

import numpy as np
import astropy.units as u 


""" System Instance
__init__: initialize Body with parameters
vec: concatenate x, y, z and vx, vy, vz vector
"""
class Body():
    

    """Initialize Body object instance
    """
    def __init__(self, mass, x_vec, v_vec, charge=None):
        
        if isinstance(mass, u.Quantity):
            self.mass = mass.si.value
        else:
            self.mass = mass

        if isinstance(x_vec[0], u.Quantity):
            self.x_vec = np.array([i.si.value for i in x_vec])    
        else:
            self.x_vec = np.array(x_vec)
        
        if isinstance(v_vec[0], u.Quantity):
            self.x_vec = np.array([i.si.value for i in v_vec])    
        else:
            self.x_vec = np.array(v_vec)

        if isinstance(charge, u.Quantity):
            self.charge = charge.si.value
        else:
            self.charge = charge


    """Store pos (x, y, z) and vel (vx, vy, vz) in array
    """       
    def vec(self):
        return np.concatenate((self.x_vec, self.v_vec))