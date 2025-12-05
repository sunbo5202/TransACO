# TransACO
This repository contains the official code from "TransACO: Transformer-Guided Ant Colony Algorithm for Combinatorial Optimization".
- [2025/11/23]: TransACO is currently under review for TETCI.
## Abstract
Ant Colony Optimization (ACO) is a heuristic algorithm that has been widely used to solve combinatorial optimization problems. However, determining heuristic measures and initial pheromones levels in an ACO is problem-specific and often requires expert intervention, posing a significant challenge for the users. We investigate how the Transformer, a powerful learning model based on a self-attention mechanism, can be used to enable the automatic design of heuristic measures and pheromones. We present Transformer-Guided ACO (**_TransACO_**), a framework that combines neural methods with heuristic algorithms. The Transformer is designed as a learner that encodes problem instances, captures individual interactions, and generates heuristic measures and pheromone matrix. Additionally, a candidate point perturbation local search technique is proposed to enhance exploration. We conduct extensive experiments on four representative combinatorial optimization problems—traveling salesman problem, capacitated vehicle routing problem, orienteering problem, and multiple knapsack problem—across diverse instance sizes. Notably, in cross-scale generalization experiments, a model trained solely on small instances successfully transfers to 10k-node problems: on TSP-10k, TransACO achieves a 4.12\% improvement, and on CVRP-10k, it yields a 1.16\% gain over strong baselines. These findings demonstrate the robustness, adaptability, and scalability of TransACO.

**The motivation of TransACO framework:**
<div align="center">
<table>
<tr>
    <td align="center" width="10%"><img src="https://github.com/sunbo5202/TransACO/blob/main/Fig/Motivation.png" 
        alt="motivation"/>
    </td>
    <td align="left" width="10%">
        (a) Hybrid methods commonly require substantial expert knowledge to design initial pheromones and heuristic measures independently.
        <br/>   <br/>     
        (b) TransACO automates the design of pheromone matrix and heuristic measures by learning correlated features within problem instances.
    </td>
</tr>
</table>
</div>

**The overview of TransACO framework:**
<div align="center">
<table>
<tr>
    <td align="left" width="10%">
        Taking TSP as an example, the method calculates the distance matrix and node coordinates based on the problem instance. The results are then input to the Transformer, which generates the HM and PM. ACO constructs the initial solution and optimizes it using optional local search techniques. Finally, the method samples from the obtained solution and calculates the reward as feedback.
        <br/>   <br/>
        In modified Transformer, we depart from the standard query–key–value formulation and adopt a query–key-only variant that is tailored to constructing the PM and HM. 
For each city $i$ in a TSP instance, the Transformer encoder outputs a shared query vector $q_i \in \mathbb{R}^d$ and two type-specific key vectors $k^{\text{p}}_i, k^{\text{h}}_i \in \mathbb{R}^d$ after linear projection and a ReLU nonlinearity.
        <br/>   <br/>
        In ACO, we generate the transition probability by leveraging the cooperation between the HM and PM. After constructing a solution, we further apply a candidate-solution perturbation mechanism to perform local search. 
    </td>
    <td align="center" width="10%"><img src="https://github.com/sunbo5202/TransACO/blob/main/Fig/Framework.png" 
        alt="motivation"/>
    </td>
</tr>
</table>
</div>

**Training process of the Transformer-based learner:**
<div align=center>
    <img src="https://github.com/sunbo5202/TransACO/blob/main/Fig/Train.png" 
        alt="framework" width="55%"/>
</div>
The green box represents the ACO solver, and the red box the learner. Firstly, the ACO solver solves the problem instance based on heuristic measures and initial pheromones. Secondly, we take the solutions obtained by the population as samples and calculate rewards based on the average and optimal values. Finally, the Transformer-based learner is updated.

## 🔑 Repository requirements
- Create python environment using conda:
```shell
conda create -n transaco-py38 python=3.8 -y
conda activate transaco-py38
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install d2l networkx==2.8.4 numpy==1.23.3 numba==0.56.4
```
- PyTorch Scatter / Sparse / Geometric are strongly dependent on the PyTorch and CUDA versions; if using PyTorch 1.7.0 with CUDA 11.0, they can be installed with the following commands.
```shell
pip install torch-scatter==2.0.7 -f https://data.pyg.org/whl/torch-1.7.0+cu110.html
pip install torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.7.0+cu110.html
pip install torch-geometric==2.0.4
```
## Development
### Setting up
- First, for each problem type, run the utils.py file in the corresponding problem folder to generate the training, validation, and test datasets.
- Then, run train.py to train the model and test.py to evaluate it on the test set.
```shell
python train.py --nodes 100
python test.py --nodes 100
```

