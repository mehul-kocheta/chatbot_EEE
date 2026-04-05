import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_results():
    try:
        with open("benchmark_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("benchmark_results.json not found.")
        return

    # Group by difficulty
    # We'll use a specific order for the x-axis
    difficulty_order = ["simple", "medium", "complex"]
    
    stats = {diff: {"agent_times": [], "gpt_times": [], "agent_success": 0, "total": 0} for diff in difficulty_order}
    
    for r in results:
        diff = r["difficulty"]
        if diff not in stats: continue
        
        stats[diff]["total"] += 1
        
        # Success check (simplified: no "Agent Error" in the output)
        if "Agent Error" not in r["agent_answer"]:
            stats[diff]["agent_success"] += 1
            stats[diff]["agent_times"].append(r["agent_time_seconds"])
        
        # GPT times are always valid in our script
        stats[diff]["gpt_times"].append(r["gpt_time_seconds"])

    # Calculate averages
    avg_agent_times = []
    avg_gpt_times = []
    success_rates = []
    
    for diff in difficulty_order:
        s = stats[diff]
        # Use median or mean? Mean is fine for a general trend
        a_times = s["agent_times"]
        g_times = s["gpt_times"]
        
        avg_agent_times.append(np.mean(a_times) if a_times else 0)
        avg_gpt_times.append(np.mean(g_times) if g_times else 0)
        success_rates.append(s["agent_success"] / s["total"] * 100 if s["total"] > 0 else 0)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Average Execution Time
    x = np.arange(len(difficulty_order))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, avg_gpt_times, width, label='GPT-5.4 (Pure LLM)', color='#4285F4')
    rects2 = ax1.bar(x + width/2, avg_agent_times, width, label='Agent (MATLAB Tool)', color='#EA4335')

    ax1.set_ylabel('Average Time (seconds)')
    ax1.set_title('Execution Time by Complexity')
    ax1.set_xticks(x)
    ax1.set_xticklabels([d.capitalize() for d in difficulty_order])
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Plot 2: Time Difference (Agent - GPT)
    time_diff = np.array(avg_agent_times) - np.array(avg_gpt_times)
    ax2.plot(difficulty_order, time_diff, marker='o', linestyle='-', color='#FBBC05', linewidth=2, markersize=8)
    ax2.set_ylabel('Time Overhead Difference (Agent - GPT) [s]')
    ax2.set_title('Inefficiency Gap vs Complexity')
    ax2.set_xticklabels([d.capitalize() for d in difficulty_order])
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line or labels
    for i, txt in enumerate(time_diff):
        ax2.annotate(f"{txt:.1f}s", (difficulty_order[i], time_diff[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.savefig("benchmark_analysis.png")
    print("Plot saved to benchmark_analysis.png")
    
    # Also print a small table for summary
    print("\nBenchmark Summary:")
    print(f"{'Difficulty':<12} | {'Avg GPT T':<10} | {'Avg Agent T':<11} | {'Gap (A-G)':<10} | {'Agent Success':<13}")
    print("-" * 65)
    for i, diff in enumerate(difficulty_order):
        gap = avg_agent_times[i] - avg_gpt_times[i]
        print(f"{diff.capitalize():<12} | {avg_gpt_times[i]:>9.2f}s | {avg_agent_times[i]:>10.2f}s | {gap:>9.2f}s | {success_rates[i]:>12.1f}%")

if __name__ == "__main__":
    plot_results()
