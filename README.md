## Koopman-for-Rimless-Wheel
This repository includes a portion of the code for the paper "Koopman global linearization of contact dynamics for robot locomotion and manipulation enables elaborate control"

## Abstract
Controlling robots that dynamically engage in contact with their environment is a pressing challenge. Whether a legged robot making-and-breaking contact with a floor, or a manipulator grasping objects, contact is everywhere. Unfortunately, the switching of dynamics at contact boundaries makes control difficult. Predictive controllers face non-convex optimization problems when contact is involved. Here, we overcome this difficulty by applying Koopman operators to subsume the segmented dynamics due to contact changes into a unified, globally-linear model in an embedding space. We show that viscoelastic contact at robot-environment interactions underpins the use of Koopman operators without approximation to control inputs. This methodology enables the convex Model Predictive Control of a legged robot, and the real-time control of a manipulator engaged in dynamic pushing. In this work, we show that our method allows robots to discover elaborate control strategies in real-time over time horizons with multiple contact changes, and the method is applicable to broad fields beyond robotics. 

## Code Structure
The code is structured into four folders:
1. MATLAB_Rimless_Wheel: Code for data generation and animation of the actuated rimless wheel
2. Python_Rimless_Wheel: Code for Koopman model generation and simulation of the actuated rimless wheel
3. Data_Rimless_Wheel: Saved Koopman models, reference trajectories, and datasets used to produce the results in the paper.
4. MATLAB_Cart-Pole_Walls: All scripts for cart-pole system with compliant walls


## How to Use

Recommended to go to main result (Python_Rimless_Wheel/rimless_wheel_simulation.py) to see K-MPC controlling the Rimless Wheel


Generate Data: MATLAB_Rimless_Wheel\generate_data\generate_data_3_spoke.m

Creates nonlinear simulation data of the Rimless Wheel (only three actuated spokes)

Outputs data files to "Data_Rimless_Wheel\training"


Create Koopman Model: Python_Rimless_Wheel\make_model\make_cck_model.py or Python_Rimless_Wheel\make_model\make_dmdc_model.py

Uses premade dataset to generate either a CCK or DMDc model

Outputs model data to "Data_Rimless_Wheel\koopman_models"


Main Result: Python_Rimless_Wheel/rimless_wheel_simulation.py

Uses premade CCK model (with embedding compensation) to control a rimless wheel that is initialized on the reference trajectory. 

Change the model type (CCK, DMDc, LL) on line 389. Turn off/on embedding compensation on line 390

Can also change the time horizon (line 369), total simulation length (line 356), initial condition (line 391 and line 454)

Script saves simulation data to "Data_Rimless_Wheel\sim_results". If nothing is changed, output files "XXXX.csv" should match files labeled "XXXX_expected_result.csv"

Time on my machine: 4.9 min


Animate Results: MATLAB_Rimless_Wheel\animate\animate_from_plot.m

Simple animation of simulation data "Data_Rimless_Wheel\sim_results\z_out"

Currently plots premade data of CCK (with embedding compensation, N = 20, tf = 20, trial 31)

Saves video as MP4. If nothing is changed, output video should look identical to "expected_animation_output.mp4"


Cart-Pole Simulation:

Please refer to the README in the "MATLAB_Cart-Pole_Walls" folder



## Version Information

MATLAB				23.2.0.2428915 (R2023b)
Control System Toolbox		Version 23.2
Symbolic Math Toolbox		Version 23.2

python 				3.10.12
pandas 				1.5.3
numpy				1.26.3
scipy 				1.11.4
gurobipy 			11.0.1
matplotlib 			3.8.2
sklearn 			1.4.0

Operating System: Linux (WSL2 on Windows 11 Version 25H2) 
Kernel: 6.6.87.2-microsoft-standard-WSL2


