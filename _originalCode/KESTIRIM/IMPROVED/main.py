# main.py
# Main execution script for running the shock solver.

import yaml
import logging
import pandas as pd
from numpy import sqrt
from shock_solver_module import FlowState, ShockSolver

def main():
    """Main workflow for the shock simulation."""
    # --- 1. Load Configuration and Setup Logging ---
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please create it.")
        return
        
    log_file = config['output_settings']['log_file']
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'), # Overwrite log file each run
            logging.StreamHandler()
        ]
    )

    logging.info("--- Starting High-Temperature Shock Simulation ---")
    logging.info(f"Configuration loaded from config.yaml")

    # --- 2. Initialize Solver and Define Initial Conditions ---
    gas_model_path = config['model_settings']['gas_yaml']
    solver = ShockSolver(gas_yaml_path=gas_model_path)
    
    # Define initial conditions based on your test case
    M1 = 28.0
    T1 = 283.0
    P1 = 48.13
    # Use perfect gas relations to find initial speed of sound and density
    R = 287.1 # Specific gas constant for air
    gamma = 1.4
    a1 = sqrt(gamma * R * T1)
    u1 = M1 * a1
    rho1 = P1 / (R * T1)

    initial_state = FlowState(M=M1, u=u1, P=P1, T=T1, rho=rho1, gas_model_name=gas_model_path)
    logging.info(f"Initial Upstream State: {initial_state}")

    # --- 3. Run the Solver ---
    logging.info("Calling nshock solver...")
    solver_params = config['solver_settings']
    final_state = solver.nshock(initial_state, solver_params)

    if final_state is None:
        logging.error("Solver failed to find a solution.")
        return
        
    logging.info(f"Final Downstream State: {final_state}")

    # --- 4. Save Results ---
    results_df = pd.DataFrame({
        'Condition': ['Upstream', 'Downstream'],
        **{k: [i, f] for k, i, f in zip(initial_state.to_dict().keys(), initial_state.to_dict().values(), final_state.to_dict().values())}
    })
    
    output_path = config['output_settings']['results_file']
    results_df.to_csv(output_path, index=False)
    logging.info(f"Results successfully saved to {output_path}")
    logging.info("--- Simulation Finished ---")

if __name__ == "__main__":
    main()