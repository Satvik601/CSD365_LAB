
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Aircraft speeds in metres per second
VB = 200
VF = 300

# Differential equations
def equations(t, state):
    R, theta = state

    # Avoid division by zero
    R = max(R, 1e-6)

    dR_dt = VB * np.cos(theta) - VF
    dtheta_dt = -VB * np.sin(theta) / R

    return [dR_dt, dtheta_dt]


# Stop simulation when distance reaches 1 cm
def hit_target(t, state):
    return state[0] - 0.01

hit_target.terminal = True
hit_target.direction = -1


# Initial distances and angles
initial_distances = [2000, 5000, 8000]
initial_angles = [15, 45, 75]

results = []

# Plot distance against time for all simulations
plt.figure(figsize=(10, 6))

for R0 in initial_distances:
    for angle in initial_angles:

        theta0 = np.radians(angle)

        solution = solve_ivp(
            equations,
            (0, 300),
            [R0, theta0],
            events=hit_target,
            max_step=0.1,
            rtol=1e-8,
            atol=1e-10
        )

        # Calculate interception time
        if len(solution.t_events[0]) > 0:
            hit_time = solution.t_events[0][0]
            results.append((R0, angle, hit_time))
        else:
            hit_time = None
            results.append((R0, angle, None))

        # Plot distance for this simulation
        label = f"R0={R0} m, angle={angle} deg"
        plt.plot(solution.t, solution.y[0], label=label)


# Print results table
print("\nInterception Results")
print("-" * 48)
print(f"{'Distance (m)':<16}{'Angle (deg)':<15}{'Time (s)'}")
print("-" * 48)

for R0, angle, hit_time in results:
    if hit_time is None:
        print(f"{R0:<16}{angle:<15}No hit")
    else:
        print(f"{R0:<16}{angle:<15}{hit_time:.2f}")


# Configure the graph
plt.axhline(0, linestyle="--", color="black")
plt.xlabel("Time (seconds)")
plt.ylabel("Distance between aircraft (metres)")
plt.title("Pure Pursuit: Distance vs Time")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()
