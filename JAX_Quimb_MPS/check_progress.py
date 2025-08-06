import os
import re
from collections import defaultdict

def check_simulation_progress():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Checkpoint_Metrcs")
    
    # Ensure the base directory exists
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found!")
        return
    
    print(f"Scanning {base_dir} for simulation checkpoints...\n")
    print("=" * 80)
    print(f"{'Simulation Name':<50} {'Latest Epoch':<15} {'Status':<15}")
    print("=" * 80)
    
    simulations = defaultdict(int)
    
    # Iterate through all simulation directories
    for sim_dir in sorted(os.listdir(base_dir)):
        full_sim_dir = os.path.join(base_dir, sim_dir)
        
        # Skip if not a directory
        if not os.path.isdir(full_sim_dir):
            continue
        
        # Check for checkpoints directory
        checkpoints_dir = os.path.join(full_sim_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            simulations[sim_dir] = (-1, "No checkpoints")
            continue
            
        # Find all checkpoint files
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("checkpoint_")]
        
        if not checkpoint_files:
            simulations[sim_dir] = (0, "Empty directory")
            continue
            
        # Extract epoch numbers using regex
        latest_epoch = 0
        for checkpoint in checkpoint_files:
            match = re.search(r'checkpoint_(\d+)', checkpoint)
            if match:
                epoch = int(match.group(1))
                latest_epoch = max(latest_epoch, epoch)
        
        simulations[sim_dir] = (latest_epoch, "Running" if latest_epoch< 7000 else "Done")
    
    # Print results
    for sim_name, (epoch, status) in sorted(simulations.items(), key=lambda x: x[1][0], reverse=True):
        epoch_str = str(epoch) if epoch >= 0 else "N/A"
        print(f"{sim_name:<50} {epoch_str:<15} {status:<15}")
    
    print("=" * 80)
    print(f"Found {len(simulations)} simulation directories")

if __name__ == "__main__":
    check_simulation_progress()
