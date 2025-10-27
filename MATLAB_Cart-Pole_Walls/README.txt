generate_data.m:
- run to generate data from N trajectories
- saves to data/training folder

generate_koopman_controller.m:
- run to generate a Koopman model and Koopman LQR gains
- evaluates control performance of many models/model parameters to optimize RBF dilation (epsilon) and number of observables
- saves to koopman_models folder

simulate_cartpole.m:
- run to simulate and animate cart-pole with given controller type
- controller_type = {0 passive, 2 K-LQR, 3 LQR}

compare_controller.m:
- compares performance of K-LQR and LQR for num_tests trajectories


