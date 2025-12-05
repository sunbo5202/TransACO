# TransACO
This repository contains the official code from "TransACO: Transformer-Guided Ant Colony Algorithm for Combinatorial Optimization".
- [2025/11/23]: TransACO is currently under review for TETCI.
## Abstract
Ant Colony Optimization (ACO) is a heuristic algorithm that has been widely used to solve combinatorial optimization problems. However, determining heuristic measures and initial pheromones levels in an ACO is problem-specific and often requires expert intervention, posing a significant challenge for the users. We investigate how the Transformer, a powerful learning model based on a self-attention mechanism, can be used to enable the automatic design of heuristic measures and pheromones. We present Transformer-Guided ACO (TransACO), a framework that combines neural methods with heuristic algorithms. The Transformer is designed as a learner that encodes problem instances, captures individual interactions, and generates heuristic measures and pheromone matrix. Additionally, a candidate point perturbation local search technique is proposed to enhance exploration. We conduct extensive experiments on four representative combinatorial optimization problems—traveling salesman problem, capacitated vehicle routing problem, orienteering problem, and multiple knapsack problem—across diverse instance sizes. Notably, in cross-scale generalization experiments, a model trained solely on small instances successfully transfers to 10k-node problems: on TSP-10k, TransACO achieves a 4.12\% improvement, and on CVRP-10k, it yields a 1.16\% gain over strong baselines. These findings demonstrate the robustness, adaptability, and scalability of TransACO.
## 🔑 Usage

### Dependencies

- Python 3.8
- CUDA 11.0
- PyTorch 1.7.0
- [PyTorch Scatter](https://github.com/rusty1s/pytorch_scatter) 2.0.7
- [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) 0.6.9
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) 2.0.4
- d2l
- [networkx](https://networkx.org/) 2.8.4
- [numpy](https://numpy.org/) 1.23.3
- [numba](https://numba.pydata.org/) 0.56.4
```
