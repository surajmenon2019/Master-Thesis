import numpy as np
import matplotlib.pyplot as plt
import os

# --- CONFIGURATION ---
METHODS = [
    "Cold Start", 
    "Flow Only", 
    "Warm Start (ForwardKL)", 
    "Warm Start (ReverseKL)", 
    "SVGD",
    "MDN"
]

# Set a color map to ensure distinct colors for each method
COLORS = {
    "Cold Start": "tab:blue",
    "Flow Only": "tab:purple",
    "Warm Start (ForwardKL)": "tab:orange",
    "Warm Start (ReverseKL)": "tab:green",
    "SVGD": "tab:red",
    "MDN": "tab:brown"
}

def smooth(data, window=10):
    """
    Smoothes data using a moving average that PRESERVES array length.
    This ensures Episode 100 stays at x=100.
    """
    if len(data) < window:
        return data
    
    # Create a kernel (window)
    kernel = np.ones(window) / window
    
    # "same" mode keeps the output size equal to input size by padding
    smoothed = np.convolve(data, kernel, mode='same')
    
    # The edges will be inaccurate with zero-padding, so we fix the start/end
    # (Optional refinement for very neat plots, but 'same' is usually sufficient for trends)
    return smoothed

def plot_rewards():
    plt.figure(figsize=(12, 7))
    found_any = False
    
    # Get standard style
    #plt.style.use('seaborn-v0_8-whitegrid') # Optional: Makes it look academic
    
    for method in METHODS:
        filename = f"rewards_{method}.npy"
        
        if os.path.exists(filename):
            print(f"Loading {filename}...")
            try:
                rewards = np.load(filename)
            except:
                print(f"!! Error loading {filename}. Skipping.")
                continue

            # Check if empty
            if len(rewards) == 0:
                print(f"!! {filename} is empty.")
                continue
                
            x_axis = np.arange(len(rewards)) # Explicit x-axis
            
            # 1. Plot RAW data (Faint, thin line)
            # This proves "Accuracy" - showing the real variance
            c = COLORS.get(method, None)
            plt.plot(x_axis, rewards, color=c, alpha=0.25, linewidth=1)
            
            # 2. Plot SMOOTHED data (Solid, thick line)
            # This shows the "Convergence Trend"
            smoothed_rewards = smooth(rewards, window=10)
            plt.plot(x_axis, smoothed_rewards, label=method, color=c, linewidth=2.5)
            
            found_any = True
        else:
            # Only print if you expected it to be there
            pass

    if not found_any:
        print("\n[!] No reward files found.")
        print("    Run 'train_safety_agents.py' first to generate .npy files.")
        return

    # Formatting for Thesis
    plt.title("Policy Convergence: Warm Start vs Baselines", fontsize=14, pad=15)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Total Reward (Reward - Cost)", fontsize=12)
    plt.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=10)
    plt.xlim(left=0)
    
    # Save
    save_path = "thesis_policy_convergence_accurate.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n>>> SUCCESS. High-res plot saved to '{save_path}'")
    plt.show()

if __name__ == "__main__":
    plot_rewards()