import json
import matplotlib.pyplot as plt
import numpy as np

def visualize_saved_routes(json_path="routes_data.json"):
    # 1. Load visualization payload
    try:
        with open(json_path, "r") as f:
            routes_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Could not find '{json_path}'. Make sure you ran the benchmark script first.")
        return

    color_scheme = {
        "Amazon_Actual": "#7f7f7f",    # Muted Gray
        "WS_LR_QAOA_Ideal": "#1f77b4", # Clean Blue
        "WS_LR_QAOA_Noisy": "#ff7f0e"  # Vivid Orange
    }

    # 2. Iterate through each route in the payload
    for data in routes_data:
        route_id = data["route_id"]
        coords = np.array(data["coords"])
        solutions = data["solutions"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
        fig.suptitle(f"Routing Performance — {route_id}", fontsize=16, fontweight="bold")

        for ax, (sol_name, sol_info) in zip(axes, solutions.items()):
            route = sol_info["route"]
            cost = sol_info["cost"]

            # Plot Depot (Stop 0) and Stops
            ax.scatter(coords[0, 0], coords[0, 1], c="red", s=150, zorder=5, label="Depot (0)", marker="s")
            ax.scatter(coords[1:, 0], coords[1:, 1], c="black", s=70, zorder=4)

            # Annotate stop numbers
            for idx, (x, y) in enumerate(coords):
                ax.annotate(str(idx), (x + 0.3, y + 0.3), fontsize=10, fontweight="bold")

            # Plot path loop
            full_path = route + [route[0]]
            path_coords = coords[full_path]
            
            ax.plot(
                path_coords[:, 0], 
                path_coords[:, 1], 
                color=color_scheme.get(sol_name, "#2ca02c"),
                linestyle="-", 
                linewidth=2.5, 
                label=f"Cost: {cost:.1f} min"
            )

            ax.set_title(f"{sol_name.replace('_', ' ')}", fontsize=13, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(loc="upper right")
            ax.set_xlabel("X Distance (km)")
            if ax == axes[0]:
                ax.set_ylabel("Y Distance (km)")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_saved_routes("routes_data.json")