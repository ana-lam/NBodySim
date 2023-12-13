#!/usr/bin/env python3

from simulation.body import Body
from simulation.system import System
from utils.animation import create_animation
import argparse
from simulation.forces import gravitational_force, electrostatic_force
import multiprocessing as mp

def parse_bodies(num_bodies, gravity, coulomb=None, masses=None, x_vecs=None, v_vecs=None, charges=None):

    bodies = []
    for i in range(num_bodies):
        try:
            if masses is not None:
                mass = masses[i]
            else:
                mass = float(input(f"Enter mass for body {i + 1}: "))
            if x_vecs is not None:
                x_vec = x_vecs[i * 3: (i + 1) * 3]
            else:
                x_vec = [float(x) for x in input(f"Enter position coordinates for body {i + 1} (x y z format): ").split()]
            if v_vecs is not None:
                v_vec = v_vecs[i * 3: (i + 1) * 3]
            else:
                v_vec = [float(vx) for vx in input(f"Enter velocity for body {i + 1} (v_x v_y v_z format): ").split()]
            if coulomb:
                if charges is not None:
                    charge = charges[i]
                else:
                    charge = float(input(f"Enter charge for body {i + 1}: "))
        except ValueError:
            print("ValueError: Please enter a valid numerical value.")
            return None
        
        if coulomb:
            bodies.append(Body(mass, x_vec, v_vec, charge))
        else:
            bodies.append(Body(mass, x_vec, v_vec))
    
    return bodies

def main():

    parser = argparse.ArgumentParser(description='Run N Body Simulations using numerical integration methods of choice & animate them.')
    parser.add_argument("--num-bodies", type=int, default=1, help="Number of bodies in the simulation (simulation)")
    parser.add_argument("--gravity", action='store_true', default=True, help="Model gravitational forces (simulation)")
    parser.add_argument("--coulomb", action='store_true', help="Model electrostatic forces (simulation)")
    parser.add_argument("--dimensionless", action='store_true', default=False, help="Dimensionless, constants=1 (simulation)")

    parser.add_argument("--mass", type=float, nargs="+", help="Mass of the bodies (simulation)")
    parser.add_argument("--x-vec", type=float, nargs="+", help="Position vector (x, y, z) of the bodies (simulation)")
    parser.add_argument("--v-vec", type=float, nargs="+", help="Velocity vector (vx, vy, vz) of the bodies (simulation)")
    parser.add_argument("--charge", type=float, nargs="+", help="Charge of the bodies (simulation)")
    parser.add_argument("--int", choices=["RK4", "leapfrog", "velocity_verlet", "yoshida"], default="RK4", help="Integration method (simulation)")
    parser.add_argument("--T", type=float, nargs="?", help="Total time to run simulation (simulation)")
    parser.add_argument("--dt", type=float, nargs="?", help="Timestep size (simulation)")
    
    parser.add_argument("--file-name", type=str, default="output", help="Name for mp4 file (animation)")
    parser.add_argument("--skip-frames", type=float, nargs="?", help="Skip frames (animation)", default=None)
    parser.add_argument("--fps", type=float, nargs="?", help="Frames per second (animation)", default=30)


    args = parser.parse_args()

    bodies = parse_bodies(args.num_bodies, args.gravity, args.coulomb, args.mass, args.x_vec, args.v_vec, args.charge)

    print("\n")
    if bodies is not None:
        print("Initialized the following bodies:")
        for i, body in enumerate(bodies):
            if args.coulomb:
                print(f"Body {i + 1} - mass: {body.mass}, x_vec: {body.x_vec}, v_vec: {body.v_vec}, charge: {body.charge}")
            else:
                print(f"Body {i + 1} - mass: {body.mass}, x_vec: {body.x_vec}, v_vec: {body.v_vec}")

    if args.dimensionless:
        simulation = System(bodies, dimensionless=True)
        if args.coulomb:
            simulation.ODEs_to_solve(electrostatic_force)
        else:
            simulation.ODEs_to_solve(gravitational_force)
    else:
        simulation = System(bodies, dimensionless=False)
        if args.coulomb:
            simulation.ODEs_to_solve(electrostatic_force)
        else:
            simulation.ODEs_to_solve(gravitational_force)

    if args.int is not None:
        integrator = args.int
    else:
        print("\n")
        integrator = input("Enter integrator of choice (RK4, leapfrog, velocity_verlet, yoshida): ")
    
    if integrator not in ['RK4', 'leapfrog', 'velocity_verlet', 'yoshida']:
        raise ValueError('ValueError: Not a valid integrator (choose from RK4, leapfrog, velocity_verlet, yoshida).')
        return None
    
    simulation.method_to_use(integrator)

    try:
        if args.T is not None:
            T = args.T
        else:
            T = float(input("Enter total time to run sim: "))
        if args.dt is not None:
            dt = args.dt
        else:
            dt = float(input("Enter timestep size: "))
    except ValueError:
            print("ValueError: Please enter a valid numerical value.")
            return None

    if args.skip_frames:
        skip_frames = args.skip_frames
    else:
        skip_frames=None
    if args.fps:
        fps = args.fps
    
    print("\n")
    print("Now we will begin time evolving!!")
    simulation.evolve(T, dt)
    print("\n")
    create_animation(simulation, output_dir="output", file_name=args.file_name, potential_on=True, fps=fps, skip_frames=skip_frames)

if __name__ == "__main__":

    from utils.animation import create_animation
    from utils.visualization import render_positions, render_positions_w_potential

    main()