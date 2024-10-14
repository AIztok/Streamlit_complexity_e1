import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Add description at the beginning
st.title("System Failure Simulation")

st.write("""
This simulation model calculates the probability of complete system failure based on random dependencies between variables. 
Each variable represents an element in a system, and each element can either succeed or fail based on its random value and the influence of dependent elements.

### Assumptions:
- **Random variables**: Each variable is assigned a random value between 0 and 1, representing its state.
- **Threshold for failure**: A variable fails if its value is below a defined threshold (set to 0.05 by default).
- **Dependencies**: Variables can influence each other. If a variable fails, its dependent variables may also fail, propagating failure through the system.
- **Complete system failure**: If all variables in a simulation fall below the failure threshold, the system is considered to have failed.
- **Connections**: The number of dependencies between variables is controlled by the "Number of Connections" slider.

### Goal:
Simulate multiple runs to see how often complete system failure occurs out of the total number of simulations. Each dot in the visualized grid represents a simulation, turning **red** if the system fails, and **green** if it succeeds.
""")

def create_random_vector(n):
    return np.random.rand(n, 1)

def create_dependency_matrix(n, C):
    dependency_matrix = np.zeros((n, n))
    possible_positions = [(i, j) for i in range(n) for j in range(n) if i != j]
    selected_positions = np.random.choice(len(possible_positions), C, replace=False)
    
    for pos in selected_positions:
        i, j = possible_positions[pos]
        dependency_matrix[i, j] = 1

    return dependency_matrix

# Visualization function with adjusted positions
def plot_simulation_results(results, A):
    rows = int(np.sqrt(A))
    cols = A // rows + (A % rows > 0)
    radius = 0.4
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')

    # Plot results as dots, red for failures and green for successes
    for idx, result in enumerate(results):
        x = (idx % cols) + radius   # Shift right by radius/2
        y = (idx // cols) + radius   # Shift up by radius/2
        color = 'red' if result else 'green'
        ax.add_patch(plt.Circle((x, y), radius, color=color))

    ax.set_xticks([])
    ax.set_yticks([])
    st.pyplot(fig)


# Use Streamlit's sliders to get user inputs
A = st.slider("Number of simulations", 1, 1000, step=100)  # Step size of 100
n = st.slider("Number of variables", 2, 500, step=100)  # Step size of 10
max_connections = n * (n - 1)
C = st.slider("Number of connections", 0, max_connections, step=10)  # Step size of 10
threshold = 0.05

# Wrap computations in a function
@st.cache_data()
def calculate_system_failures(A, n, C):
    results = []
    num_system_failures = 0
    
    for i in range(A):
        dependency_matrix = create_dependency_matrix(n, C)
        random_vector = create_random_vector(n)
        
        num_below_threshold = 0
        for row_idx, row in enumerate(random_vector):
            if row < threshold:
                for dep_idx, dep_val in enumerate(dependency_matrix[row_idx]):
                    if dep_val > 0 and random_vector[dep_idx] > threshold:
                        random_vector[dep_idx] = random_vector[row_idx]

        num_below_threshold = np.sum(random_vector < threshold)
        if num_below_threshold == n:
            num_system_failures += 1
            results.append(1)  # Failure
        else:
            results.append(0)  # Success
        
    return num_system_failures, results

# Run the simulation and get results
num_system_failures, simulation_results = calculate_system_failures(A, n, C)

# Display the simulation grid
plot_simulation_results(simulation_results, A)

# Show results
st.write(f"Number of system failures out of {A} simulations: {num_system_failures}")
