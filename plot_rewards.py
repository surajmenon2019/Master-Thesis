import numpy as np
import matplotlib.pyplot as plt
import os

# The methods you want to compare (Must match the names in train_mbrl.py)
METHODS = [
    "Cold Start", 
    "Flow Only", 
    "Warm Start (ForwardKL)", 
    "Warm Start (ReverseKL)", 
    "SVGD",
    "MDN"
]

def smooth(data, window=5):
    """Smoothes the reward curve for cleaner visualization."""
    if len(data) < window: return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def plot_rewards():
    plt.figure(figsize=(10, 6))
    
    found_any = False
    
    for method in METHODS:
        filename = f"rewards_{method}.npy"
        
        if os.path.exists(filename):
            print(f"Loading {filename}...")
            rewards = np.load(filename)
            
            # Smooth the curve so trends are visible
            smoothed_rewards = smooth(rewards, window=10)
            
            # Plot
            plt.plot(smoothed_rewards, label=method, linewidth=2)
            found_any = True
        else:
            print(f"!! Skipped {filename} (Not found)")

    if not found_any:
        print("No reward files found! Run train_mbrl.py first.")
        return

    plt.title("Policy Convergence: Comparison of Sampling Methods")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig("thesis_policy_convergence.png", dpi=300)
    print("\n>>> Plot saved to 'thesis_policy_convergence.png'")

if __name__ == "__main__":
    plot_rewards()