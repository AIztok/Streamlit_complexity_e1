import streamlit as st
import numpy as np

def create_random_vector(n):
    return np.random.rand(n,1)

def create_dependency_matrix(n, C):
    # Generate a zero matrix of size n x n
    dependency_matrix = np.zeros((n, n))

    # Randomly place C 1's in the matrix
    positions = np.random.choice(n*n, C, replace=True)
    dependency_matrix.flat[positions] = 1

    # Make sure diagonal values are 0
    np.fill_diagonal(dependency_matrix, 0)

    # Determine number of non-zero diagonal values
    diag_nz = np.count_nonzero(np.diag(dependency_matrix))

    # Make sure diagonal values are 0
    np.fill_diagonal(dependency_matrix, 0)

    # Determine which non-diagonal positions are zero
    non_diag_positions = np.where(np.logical_and(dependency_matrix != 1, np.eye(n) != 1))

    # Randomly select diag_nz non-diagonal positions that are zero and make them 1
    non_diag_positions = list(zip(non_diag_positions[0], non_diag_positions[1]))
    selected_positions = np.random.choice(len(non_diag_positions), diag_nz, replace=False)
    for pos in selected_positions:
        i, j = non_diag_positions[pos]
        dependency_matrix[i, j] = 1

    return dependency_matrix

# Use Streamlit's sliders to get user inputs
A = st.slider("Number of simulations", 1, 1000)
n = st.slider("Number of variables", 2, 1000)
C = st.slider("Number of connections", 1, n*n)

threshold = 0.05
num_system_failures = 0

# Wrap your computations in a function
@st.cache_data()
def calculate_system_failures(A, n, C):
    num_system_failures = 0

    # Loop over simulations
    for i in range(A):
        # Create dependency matrix for this iteration
        dependency_matrix = create_dependency_matrix(n, C)
        random_vector = create_random_vector(n)
        num_below_threshold = 0

        # print(random_vector)
        # print(dependency_matrix)

        # Count number of values below the threshold
        # loop over the rows of the random_vector
        for row_idx, row in enumerate(random_vector):
            # print(row_idx) # number of the row
            # print(row) # value of the row
            # loop over the row
            for value_idx, value in enumerate(row):
                # print(value_idx)
                # print(value)
                # check if the value in the random_vector is below the treshold
                if value < threshold:
                    # Check dependencies and update random_vector accordingly
                    for dep_idx, dep_val in enumerate(dependency_matrix[row_idx]):
                        # print(dep_idx)
                        # print(dep_val)
                        if dep_val > 0 and random_vector[dep_idx] > threshold:
                            random_vector[dep_idx] = random_vector[row_idx]
                            # random_vector[row_idx] = random_vector[dep_idx]
                            #break

                    #num_below_threshold += 1
                    #print(random_vector)
        for row in random_vector:
            for value in row:
                if value < threshold:
                    num_below_threshold += 1

        if num_below_threshold == n:
            num_system_failures += 1
        
    return num_system_failures

num_system_failures = calculate_system_failures(A, n, C)

st.write(f"Number of system failures out of {A} simulations: {num_system_failures}")


#streamlit run C:\ARI\70_Complexity\Simple_Model\Complexity_simple_model_streamlit_v00.py