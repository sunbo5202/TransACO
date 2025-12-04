import time
import torch
import os
import numpy as np
from torch.distributions import Categorical, kl

from transformer_encoder import Net
from aco import ACO
from utils import *

# Import PEG functions from peg.py
from peg import (run_aco_with_peg, visualize_aco_peg, visualize_aco_peg_simple,
                 analyze_aco_peg_metrics, explain_aco_peg_behavior, 
                 reset_peg)

torch.manual_seed(1234)

EPS = 1e-10
device = 'cpu'

def infer_instance(model, prize, weight, n_ants, t_aco_diff, enable_peg=False, instance_id=0, save_peg_dir=None):
    """
    Infer instance with optional PEG visualization.
    
    Args:
        enable_peg: Whether to enable PEG logging and visualization
        instance_id: Instance identifier for PEG
        save_peg_dir: Directory to save PEG visualizations
    """
    if model:
        model.eval()
        # Transformer model returns (pm, hm) as full matrices
        pm_mat, hm_mat = model(prize, weight)
        # Normalize heuristic matrix
        hm_mat = hm_mat / (hm_mat.min() + 1e-10) + 1e-10
        aco = ACO(
            prize=prize,
            weight=weight,
            n_ants=n_ants,
            heuristic=hm_mat,
            pheromone=pm_mat,
            device=device
            )
    else:
        aco = ACO(
            prize=prize,
            weight=weight,
            n_ants=n_ants,
            device=device
            )
    
    # Reset PEG for this instance if enabled
    if enable_peg:
        reset_peg()
    
    results = torch.zeros(size=(len(t_aco_diff),), device=device)
    cumulative_iteration = 0  # Track cumulative iteration count across multiple runs
    for i, t in enumerate(t_aco_diff):
        if enable_peg:
            # Run ACO with PEG logging, passing starting iteration number
            best_obj, _ = run_aco_with_peg(aco, t, inference=False, instance_id=instance_id, start_iteration=cumulative_iteration)
            cumulative_iteration += t  # Update cumulative iteration count
            results[i] = best_obj
        else:
            # Normal run without PEG
            best_obj, _ = aco.run(t)
            results[i] = best_obj
        
        # Visualize and analyze PEG for the last iteration (if enabled)
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
def test(dataset, model, n_ants, t_aco, enable_peg=False, save_peg_dir=None, peg_instances=None):
    """
    Test model on dataset with optional PEG visualization.
    
    Args:
        enable_peg: Whether to enable PEG logging
        save_peg_dir: Directory to save PEG visualizations
        peg_instances: List of instance indices to generate PEG for (default: first 3 instances)
    """
    if isinstance(t_aco, int):
        t_aco = [t_aco]
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),), device=device)
    
    # Default: generate PEG for first 3 instances
    if peg_instances is None:
        peg_instances = [0, 1, 2] if enable_peg else []
    
    start = time.time()
    for idx, (prize, weight) in enumerate(dataset):
        # Enable PEG for selected instances
        peg_enabled = enable_peg and idx in peg_instances
        results = infer_instance(model, prize, weight, n_ants, t_aco_diff,
                                enable_peg=peg_enabled, instance_id=idx, save_peg_dir=save_peg_dir)
        sum_results += results
    end = time.time()
    
    if enable_peg:
        print(f"\nPEG visualization generated for instances: {peg_instances}")
    
    return sum_results / len(dataset), end-start


def main(n_node, model_file=None, n_ants=20, t_aco=None, enable_peg=False, 
         peg_dir=None, num_test_instances=None, peg_instances=None):
    """
    Main test function for MKP.
    
    Args:
        n_node: Problem scale
        model_file: Path to model checkpoint (None for baseline)
        n_ants: Number of ants
        t_aco: List of iteration counts to test (or single integer)
        enable_peg: Whether to enable PEG visualization
        peg_dir: Directory to save PEG visualizations
        num_test_instances: Number of test instances to use
        peg_instances: List of instance indices to generate PEG for
    """
    if t_aco is None:
        t_aco = [10]
    if isinstance(t_aco, int):
        t_aco = [t_aco]
    
    test_list = load_test_dataset(n_node, device)
    
    # Limit number of test instances if specified
    if num_test_instances and len(test_list) > num_test_instances:
        test_list = test_list[:num_test_instances]
    
    print("problem scale:", n_node)
    if model_file:
        print("checkpoint:", model_file)
    else:
        print("baseline (no model)")
    print("number of test instances:", len(test_list))
    print("device:", device)
    print("n_ants:", n_ants)
    
    if enable_peg:
        if peg_instances is None:
            peg_instances = [0, 1, 2]  # Default: first 3 instances
        print("PEG visualization: ENABLED")
        print(f"PEG instances: {peg_instances}")
        print(f"PEG output directory: {peg_dir or './peg_outputs'}")
    
    # Test with model if provided
    if model_file and os.path.isfile(model_file):
        net_mkp = Net().to(device)
        net_mkp.load_state_dict(torch.load(model_file, map_location=device))
        avg_aco_best, duration = test(test_list, net_mkp, n_ants, t_aco,
                                      enable_peg=enable_peg, save_peg_dir=peg_dir,
                                      peg_instances=peg_instances)
        print('total duration: ', duration)
        for i, t in enumerate(t_aco):
            print("T={}, average obj. is {}.".format(t, avg_aco_best[i]))
    
    # Test baseline (no model)
    print("\n" + "="*60)
    print("Baseline (no model):")
    print("="*60)
    avg_aco_best, duration = test(test_list, None, n_ants, t_aco,
                                  enable_peg=False, save_peg_dir=None,
                                  peg_instances=None)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print("T={}, average obj. is {}.".format(t, avg_aco_best[i]))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-nodes", "--nodes", type=int, default=100, help="Problem scale")
    parser.add_argument("-m", "--model", type=str, default=None, 
                       help="Path to checkpoint file (None for baseline)")
    parser.add_argument("--enable_peg", action="store_true", 
                       help="Enable PEG (Population Evolution Graph) visualization")
    parser.add_argument("--peg_dir", type=str, default="./peg_outputs",
                       help="Directory to save PEG visualizations")
    parser.add_argument("--num_instances", type=int, default=None,
                       help="Number of test instances (default: use all)")
    parser.add_argument("--peg_instances", type=str, default="0,1,2",
                       help="Comma-separated list of instance indices for PEG visualization (default: 0,1,2)")
    parser.add_argument("--n_ants", type=int, default=50,
                       help="Number of ants (default: 20)")
    parser.add_argument("--t_aco", type=int, default=10,
                       help="Number of iterations (default: 10)")
    opt = parser.parse_args()
    
    # Parse PEG instances
    try:
        peg_instances = [int(x.strip()) for x in opt.peg_instances.split(',')]
    except:
        print(f"Warning: Invalid peg_instances format '{opt.peg_instances}', using default [0, 1, 2]")
        peg_instances = [0, 1, 2]
    
    if opt.enable_peg and not os.path.isdir(opt.peg_dir):
        os.makedirs(opt.peg_dir)
    
    # Determine model file path
    model_file = opt.model
    if model_file is None:
        # Try multiple possible paths
        possible_paths = [
            f'./pretrained/mkp/mkp{opt.nodes}.pt',  # Same directory
            f'../pretrained/mkp/mkp{opt.nodes}.pt',  # Parent directory
            f'../../pretrained/mkp/mkp{opt.nodes}.pt',  # Two levels up
        ]
        model_file = None
        for path in possible_paths:
            if os.path.isfile(path):
                model_file = path
                break
        if model_file is None:
            print(f"Model file 'mkp{opt.nodes}.pt' not found in any of the following paths:")
            for path in possible_paths:
                print(f"  - {path}")
            print("Will only run baseline.")
    
    main(opt.nodes, model_file, opt.n_ants, [opt.t_aco], 
         enable_peg=opt.enable_peg, peg_dir=opt.peg_dir,
         num_test_instances=opt.num_instances, peg_instances=peg_instances)    