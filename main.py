import os
import numpy as np
import astropy.units as u
from simulation.body import Body
from simulation.system import System
from simulation.forces import gravitational_force, electrostatic_force
from utils.energy_calc import calculate_total_energy
from utils.animation import create_animation

def main():

    Body1 = Body(
    mass=1,
    x_vec=np.array([-1.0024277970 , 0.0041695061, 0]),  # Initial position for star 1
    v_vec=np.array([0.3489048974  , 0.5306305100   , 0]),  # Initial velocity for star 1
    has_units=False
    )

    Body2 = Body(
        mass=1,
        x_vec=np.array([1.0024277970, -0.0041695061, 0]),  # Initial position for star 2
        v_vec=np.array([0.3489048974  , 0.5306305100   , 0]),  # Initial velocity for star 2
        has_units=False
    )
    Body3 = Body(
        mass=1,
        x_vec=np.array([0, 0, 0]),  # Initial position for star 2
        v_vec=np.array([-2*0.3489048974, -2*0.5306305100, 0]),  # Initial velocity for star 2
        has_units=False
    )

    bodies = [Body1, Body2, Body3]

    simulation = System(bodies, has_units=False, G_is_one=True)
    simulation.ODEs_to_solve(gravitational_force)
    simulation.method_to_use("leapfrog")

    # run simulation
    simulation.evolve(6*6.3490473929,0.03809428435)

    # Visualization and output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    create_animation(simulation, output_dir, "3body", potential_on=True)

if __name__ == "__main__":
    main()