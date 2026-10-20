
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Aircraft speeds in metres per second
VB = 200
VF = 300

def equations(t, state):
    R, theta = state

    # Avoid division by zero
    R = max(R, 1e-6)

    dR_dt = VB * np.cos(theta) - VF
    dtheta_dt = -VB * np.sin(theta) / R

    return [dR_dt, dtheta_dt]


# Initial conditions
R0 = 5000
theta0 = np.radians(45)

# Stop the simulation when the distance reaches 1 cm
def hit_target(t, state):
    return state[0] - 0.01

hit_target.terminal = True
hit_target.direction = -1

# Solve the differential equations
solution = solve_ivp(
    equations,
    (0, 300),
    [R0, theta0],
    events=hit_target,
    max_step=0.1,
    rtol=1e-8,
    atol=1e-10
)

if len(solution.t_events[0]) > 0:
    hit_time = solution.t_events[0][0]
    print(f"Interception time: {hit_time:.2f} seconds")
else:
    print("No interception within 300 seconds.")
