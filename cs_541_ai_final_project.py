# -*- coding: utf-8 -*-
"""CS 541 AI Final Project

Basic Plan:
1. Board representation
2. Conflict function
3. Generate NN dataset
4. Build neural network
5. Train neural network
6. Test neural network
7. NN-guided search
8. Run multiple trials
9. Compare A* vs. NN
10. Visualize results
"""

import random
import time
import heapq
import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ============================================================
# 1. 8-Queens helper functions
# ============================================================

N = 8


def conflicts(board):
    """
    Count the number of attacking queen pairs.
    board[col] = row
    """

    total = 0

    for c1 in range(N):
        for c2 in range(c1 + 1, N):

            r1 = board[c1]
            r2 = board[c2]

            # Same row
            if r1 == r2:
                total += 1

            # Same diagonal
            elif abs(r1 - r2) == abs(c1 - c2):
                total += 1

    return total


def is_solution(board):
    return conflicts(board) == 0


def print_board(board):

    for row in range(N):
        for col in range(N):

            if board[col] == row:
                print("Q", end=" ")
            else:
                print(".", end=" ")

        print()

    print()

# ============================================================
# 2. A* Search
# ============================================================

def astar_8queens():
    """
    A* using partial queen placements.

    State:
        tuple of row positions for queens already placed.

    Example:
        (0, 4, 7)

    means queens are placed in columns 0, 1, 2.
    """

    start = tuple()

    # priority queue stores:
    # (f, g, state)
    frontier = []

    heapq.heappush(frontier, (0, 0, start))

    nodes_expanded = 0

    while frontier:

        f, g, state = heapq.heappop(frontier)

        nodes_expanded += 1

        # Goal
        if len(state) == N:
            return list(state), nodes_expanded

        next_col = len(state)

        for row in range(N):

            valid = True

            # Check conflicts with queens already placed
            for col, existing_row in enumerate(state):

                if row == existing_row:
                    valid = False
                    break

                if abs(row - existing_row) == abs(next_col - col):
                    valid = False
                    break

            if valid:

                new_state = state + (row,)

                new_g = len(new_state)

                # Simple heuristic:
                # number of queens still left to place
                h = N - len(new_state)

                new_f = new_g + h

                heapq.heappush(
                    frontier,
                    (new_f, new_g, new_state)
                )

    return None, nodes_expanded

# ============================================================
# 3. Generate data for the neural network
# ============================================================

def generate_dataset(num_samples=20000):

    X = []
    y = []

    for _ in range(num_samples):

        board = [
            random.randint(0, N - 1)
            for _ in range(N)
        ]

        X.append(board)
        y.append(conflicts(board))

    return np.array(X), np.array(y)

# ============================================================
# 4. Train neural network
# ============================================================

def train_nn():

    print("Generating NN training data...")

    X, y = generate_dataset(20000)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=300,
        random_state=42
    )

    print("Training neural network...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(
        y_test,
        predictions
    )

    print("NN Test MSE:", mse)

    return model

# ============================================================
# 5. Neural-network-guided search
# ============================================================

def nn_guided_search(model, max_iterations=10000):
    """
    Start with a random board.

    At each iteration:
    - Generate neighboring boards
    - Use NN to predict conflicts
    - Move to the board with the lowest predicted conflicts
    """

    board = [
        random.randint(0, N - 1)
        for _ in range(N)
    ]

    states_evaluated = 0

    for iteration in range(max_iterations):

        # Check real conflicts
        if conflicts(board) == 0:
            return board, iteration, states_evaluated

        neighbors = []

        # Move one queen to another row
        for col in range(N):

            original_row = board[col]

            for new_row in range(N):

                if new_row == original_row:
                    continue

                new_board = board.copy()
                new_board[col] = new_row

                neighbors.append(new_board)

        neighbors_array = np.array(neighbors)

        # NN predicts how many conflicts each board has
        predictions = model.predict(neighbors_array)

        states_evaluated += len(neighbors)

        best_index = np.argmin(predictions)

        best_board = neighbors[best_index]

        # Occasionally make a random move to avoid local minima
        if random.random() < 0.05:
            board = random.choice(neighbors)
        else:
            board = best_board

    return board, max_iterations, states_evaluated

# ============================================================
# 6. Compare A* and Neural Network
# ============================================================

def compare_methods(model, runs=20):

    astar_times = []
    astar_nodes = []

    nn_times = []
    nn_states = []
    nn_successes = 0

    print("\nRunning comparisons...\n")

    for run in range(runs):

        # -----------------
        # A*
        # -----------------

        start_time = time.perf_counter()

        astar_solution, nodes = astar_8queens()

        end_time = time.perf_counter()

        astar_times.append(
            end_time - start_time
        )

        astar_nodes.append(nodes)

        # -----------------
        # NN
        # -----------------

        start_time = time.perf_counter()

        nn_solution, iterations, evaluated = nn_guided_search(
            model
        )

        end_time = time.perf_counter()

        nn_times.append(
            end_time - start_time
        )

        nn_states.append(evaluated)

        if is_solution(nn_solution):
            nn_successes += 1

        print(
            f"Run {run + 1}: "
            f"A* time={astar_times[-1]:.5f}s, "
            f"NN time={nn_times[-1]:.5f}s, "
            f"NN conflicts={conflicts(nn_solution)}"
        )

    print("\n================ RESULTS ================")

    print("\nA*")
    print("Success rate: 100%")
    print(
        "Average runtime:",
        np.mean(astar_times)
    )
    print(
        "Average nodes expanded:",
        np.mean(astar_nodes)
    )

    print("\nNeural Network")
    print(
        "Success rate:",
        f"{(nn_successes / runs) * 100:.1f}%"
    )
    print(
        "Average runtime:",
        np.mean(nn_times)
    )
    print(
        "Average states evaluated:",
        np.mean(nn_states)
    )

# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":

    # Train neural network
    nn_model = train_nn()

    # A* example
    print("\nA* Example Solution")

    astar_solution, astar_nodes = astar_8queens()

    print("Solution:", astar_solution)
    print("Conflicts:", conflicts(astar_solution))
    print("Nodes expanded:", astar_nodes)

    print_board(astar_solution)

    # NN example
    print("NN-Guided Example Solution")

    nn_solution, iterations, states = nn_guided_search(
        nn_model
    )

    print("Solution:", nn_solution)
    print("Conflicts:", conflicts(nn_solution))
    print("Iterations:", iterations)
    print("States evaluated:", states)

    print_board(nn_solution)

    # Compare
    compare_methods(
        nn_model,
        runs=20
    )

!apt-get update -y
!apt-get install -y texlive texlive-xetex texlive-latex-extra pandoc

!jupyter nbconvert --to pdf "/content/drive/MyDrive/Colab Notebooks/CS 541 AI Final Project" >/dev/null
