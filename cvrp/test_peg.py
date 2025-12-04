import time
import torch
import os
import numpy as np

from transformer_encoder import Net
from aco import ACO, get_subroutes
from utils import load_test_dataset
from tqdm import tqdm

# Import PEG functions from peg.py
from peg import (run_aco_with_peg, visualize_aco_peg, visualize_aco_peg_simple,
                 analyze_aco_peg_metrics, explain_aco_peg_behavior, 
                 reset_peg)

from typing import Tuple, Union, List

torch.manual_seed(1234)


EPS = 1e-10
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def validate_route(distance: torch.Tensor, demands: torch.Tensor, routes: List[torch.Tensor]) -> Tuple[bool, float]:
    length = 0.0
    valid = True
    visited = {0}
    for r in routes:
        d = demands[r].sum().item()
        if d>1.000001:
            valid = False
        length += distance[r[:-1], r[1:]].sum().item()
        for i in r:
            i = i.item()
            if i<0 or i>=distance.size(0):
                valid = False
            else:
                visited.add(i)
    if len(visited) != distance.size(0):
        valid = False
    return valid, length

@torch.no_grad()
def infer_instance(model, pyg_data, demands, distances, positions, n_ants, t_aco_diff, k_sparse=None, 
                   enable_peg=False, instance_id=0, save_peg_dir=None):
    """
    Infer instance with optional PEG visualization.
    
    Args:
        enable_peg: Whether to enable PEG logging and visualization
        instance_id: Instance identifier for PEG
        save_peg_dir: Directory to save PEG visualizations (if None, only visualize for first instance)
    """
    model.eval()
    # heu_vec = model(pyg_data)
    # heu_mat = model.reshape(pyg_data, heu_vec) + EPS
    phe_mat, heu_mat = model(positions, demands, distances, )

    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat.cpu(),
        pheromone=phe_mat.cpu(),
        demand = demands.cpu(),
        distances=distances.cpu(),
        device='cpu',
        swapstar=True,
        positions=positions.cpu(),
        inference=True,
    )
    
    # Reset PEG for this instance if enabled
    if enable_peg:
        reset_peg()
    
    results = torch.zeros(size=(len(t_aco_diff),))
    cumulative_iteration = 0  # Track cumulative iteration count across multiple runs
    for i, t in enumerate(t_aco_diff):
        if enable_peg:
            # Run ACO with PEG logging, passing starting iteration number
            best_cost = run_aco_with_peg(aco, t, inference=True, instance_id=instance_id, start_iteration=cumulative_iteration)
            cumulative_iteration += t  # Update cumulative iteration count
        else:
            # Normal run without PEG
            best_cost = aco.run(t, inference=True)
        
        path = get_subroutes(aco.shortest_path)
        valid, results[i] = validate_route(distances, demands, path)
        if valid is False:
            print("invalid solution.")
        
        # Visualize and analyze PEG for the last iteration (if enabled and instance is selected)
        if enable_peg and save_peg_dir and i == len(t_aco_diff) - 1:
            os.makedirs(save_peg_dir, exist_ok=True)
            
            # Generate hierarchical layout visualization (default)
            peg_path = os.path.join(save_peg_dir, f'peg_test_instance_{instance_id}_T{t}.png')
            visualize_aco_peg(save_path=peg_path, instance_id=instance_id, layout_type='hierarchical')
            
            # Generate simple spring layout visualization (EvoMapX-style)
            visualize_aco_peg_simple(save_path=peg_path, instance_id=instance_id)
            
            # Generate analysis report
            metrics = analyze_aco_peg_metrics(instance_id=instance_id)
            explanation = explain_aco_peg_behavior(instance_id=instance_id)
            
            # Save metrics
            metrics_path = os.path.join(save_peg_dir, f'peg_metrics_instance_{instance_id}_T{t}.txt')
            with open(metrics_path, 'w') as f:
                f.write(explanation)
                f.write("\n\nQUANTITATIVE METRICS:\n")
                f.write("=" * 80 + "\n")
                for key, value in metrics.items():
                    if isinstance(value, (list, np.ndarray)):
                        f.write(f"{key}: {value[:5] if len(value) > 5 else value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            print(f"\nPEG Analysis for instance {instance_id} saved to: {metrics_path}")
            print(explanation)
    
    return results
        
    
@torch.no_grad()
def test(dataset, model, n_ants, t_aco, k_sparse=None, enable_peg=False, save_peg_dir=None, peg_instances=None):
    """
    Test model on dataset with optional PEG visualization.
    
    Args:
        enable_peg: Whether to enable PEG logging
        save_peg_dir: Directory to save PEG visualizations
        peg_instances: List of instance indices to generate PEG for (default: first 3 instances)
    """
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),))
    
    # Default: generate PEG for first 3 instances
    if peg_instances is None:
        peg_instances = [0, 1, 2] if enable_peg else []
    
    start = time.time()
    for idx, (pyg_data, demands, distances, positions) in enumerate(tqdm(dataset)):
        # Enable PEG for selected instances
        peg_enabled = enable_peg and idx in peg_instances
        results = infer_instance(model, pyg_data, demands, distances, positions, n_ants, t_aco_diff, k_sparse,
                                enable_peg=peg_enabled, instance_id=idx, save_peg_dir=save_peg_dir)
        sum_results += results
    end = time.time()
    
    if enable_peg:
        print(f"\nPEG visualization generated for instances: {peg_instances}")
    
    return sum_results / len(dataset), end-start


def main(n_node, model_file, k_sparse = None, n_ants=50, t_aco = None, enable_peg=False, 
         peg_dir=None, num_test_instances=10, peg_instances=None):
    """
    Main test function.
    
    Args:
        num_test_instances: Number of test instances to use (default: 10)
        peg_instances: List of instance indices to generate PEG for (default: [0, 1, 2])
    """
    k_sparse = k_sparse or n_node//10
    t_aco = list(range(1, t_aco+1)) if t_aco else list(range(1,11))
    test_list = load_test_dataset(n_node, k_sparse, device)
    # test_list = load_val_dataset(n_node,  k_sparse, device)
    
    # Limit number of test instances if specified
    if num_test_instances and len(test_list) > num_test_instances:
        test_list = test_list[:num_test_instances]
    
    print("problem scale:", n_node)
    print("checkpoint:", model_file)
    print("number of test instances:", len(test_list))
    print("device:", 'cpu' if device == 'cpu' else device+"+cpu" )
    print("n_ants:", n_ants)
    
    if enable_peg:
        if peg_instances is None:
            peg_instances = [0, 1, 2]  # Default: first 3 instances
        print("PEG visualization: ENABLED")
        print(f"PEG instances: {peg_instances}")
        print(f"PEG output directory: {peg_dir or './peg_test_outputs'}")

    net_tsp = Net().to(device)
    net_tsp.load_state_dict(torch.load(model_file, map_location=device))
    avg_aco_best, duration = test(test_list, net_tsp, n_ants, t_aco, k_sparse, 
                                  enable_peg=enable_peg, save_peg_dir=peg_dir, 
                                  peg_instances=peg_instances)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print("T={}, average cost is {}.".format(t, avg_aco_best[i]))


if __name__ == "__main__":
    import argparse
    import os
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("-nodes", "--nodes", type=int, default=100, help="Problem scale")
    parser.add_argument("-m", "--model", type=str, default=None, 
                       help=f"Path to checkpoint file, default to '../pretrained/cvrp_nls/cvrp{{nodes}}-best.pt'")
    parser.add_argument("--enable_peg", action="store_true", 
                       help="Enable PEG (Population Evolution Graph) visualization")
    parser.add_argument("--peg_dir", type=str, default="./peg_test_outputs",
                       help="Directory to save PEG visualizations")
    parser.add_argument("--num_instances", type=int, default=10,
                       help="Number of test instances (default: 10)")
    parser.add_argument("--peg_instances", type=str, default="0,1,2",
                       help="Comma-separated list of instance indices for PEG visualization (default: 0,1,2)")
    parser.add_argument("-i", "--iterations", type=int, default=None, help="Iterations of ACO to run")
    opt = parser.parse_args()
    n_nodes = opt.nodes

    # Parse PEG instances
    try:
        peg_instances = [int(x.strip()) for x in opt.peg_instances.split(',')]
    except:
        print(f"Warning: Invalid peg_instances format '{opt.peg_instances}', using default [0, 1, 2]")
        peg_instances = [0, 1, 2]

    filepath = opt.model or f'../pretrained/cvrp_nls/cvrp{n_nodes}-best.pt'
    if not os.path.isfile(filepath):
        print(f"Checkpoint file '{filepath}' not found!")
        exit(1)
    
    if opt.enable_peg and not os.path.isdir(opt.peg_dir):
        os.makedirs(opt.peg_dir)
    
    main(n_nodes, filepath, enable_peg=opt.enable_peg, peg_dir=opt.peg_dir,
         num_test_instances=opt.num_instances, peg_instances=peg_instances, t_aco=opt.iterations)
