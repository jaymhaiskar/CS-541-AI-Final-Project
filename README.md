# 8-Queens: A* Search vs Neural Networks

## Overview

This project explores two different Artificial Intelligence approaches for solving the classic **8-Queens Problem**:

* **A* Search** using a custom admissible heuristic
* **Neural Network guided local search** using an MLP Regressor

The goal is to compare a traditional heuristic search algorithm with a machine learning approach on a discrete constraint-satisfaction problem.

## Technologies

* Python
* NumPy
* Scikit-learn
* MLPRegressor
* A* Search
* Heap Queue

## Neural Network

A synthetic dataset of **20,000 randomly generated board configurations** is created. Each board is labeled with the number of attacking queen pairs.

The neural network uses:

* 8 input features representing queen positions
* Hidden layers: **64 and 32 neurons**
* ReLU activation
* Adam optimizer
* 80/20 train-test split

The trained model is then used to guide a local search toward board configurations with fewer predicted conflicts.

## A* Search

The A* implementation builds the board incrementally using partial queen placements. A priority queue and heuristic function guide the search toward valid configurations while avoiding unnecessary state exploration.

## Results

| Algorithm      | Success Rate | Avg. Runtime | Avg. States Explored |
| -------------- | -----------: | -----------: | -------------------: |
| A* Search      |         100% |      0.0073s |          1,966 nodes |
| Neural Network |           5% |      1.6302s |       548,920 states |

Across 20 trials, **A* significantly outperformed the neural-network approach**. The experiment demonstrates that while neural networks can learn useful evaluation patterns, traditional search algorithms can be much more effective for structured and highly discrete problems such as 8-Queens.

## Key Takeaway

Machine learning is not always the best solution for every AI problem. For the 8-Queens problem, the structure of the search space allows A* with an appropriate heuristic to solve the problem more reliably and efficiently than the MLP-guided local search.

## Authors

**Jay Mhaiskar**

* Implemented A* search and state expansion
* Developed the heuristic approach
* Researched A* and neural-network methods
* Assisted with neural-network implementation

**Navya Sri Ambati**

* Developed the neural-network data pipeline
* Worked on NN-guided local search and stochastic escape mechanisms
* Report documentation and editing
