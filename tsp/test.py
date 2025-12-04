import time
import torch

from transformer_encoder import Net
from aco import ACO
from utils import load_test_dataset
from tqdm import tqdm

EPS = 1e-10
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

@torch.no_grad()
# def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse=None):
#     model.eval()
#     phe_mat, heu_mat = model(pyg_data) #
#     aco = ACO(
#         n_ants=n_ants,
#         heuristic=heu_mat.cpu(),
#         pheromone=phe_mat.cpu(),
#         distances=distances.cpu(),
#         device='cpu',
#         local_search='nls',
#     )
#
#     results = torch.zeros(size=(len(t_aco_diff),))
#     for i, t in enumerate(t_aco_diff):
#         best_cost = aco.run(t, inference = True)
#         results[i] = best_cost
#     return results
def infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse=None):
    model.eval()
    phe_mat, heu_mat = model(pyg_data)
    aco = ACO(
        n_ants=n_ants,
        heuristic=heu_mat.cpu(),
        pheromone=phe_mat.cpu(),
        distances=distances.cpu(),
        device='cpu',
        local_search='nls',
    )

    results = torch.zeros(size=(len(t_aco_diff),))
    for i, t in enumerate(t_aco_diff):
        best_cost, best_path = aco.run(t, inference=True)
        results[i] = best_cost

    return results, best_path, heu_mat.cpu(), phe_mat.cpu()
        
    
# @torch.no_grad()
# def test(dataset, model, n_ants, t_aco, k_sparse=None):
#     _t_aco = [0] + t_aco
#     t_aco_diff = [_t_aco[i+1]-_t_aco[i] for i in range(len(_t_aco)-1)]
#     sum_results = torch.zeros(size=(len(t_aco_diff),))
#     start = time.time()
#     for pyg_data, distances in tqdm(dataset):
#         results = infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse)
#         sum_results += results
#     end = time.time()
#
#     return sum_results / len(dataset), end-start
@torch.no_grad()
def test(dataset, model, n_ants, t_aco, k_sparse=None):
    _t_aco = [0] + t_aco
    t_aco_diff = [_t_aco[i + 1] - _t_aco[i] for i in range(len(_t_aco) - 1)]
    sum_results = torch.zeros(size=(len(t_aco_diff),))
    last_heuristic = None
    last_pheromone = None
    start = time.time()
    for pyg_data, distances in tqdm(dataset):
        results, best_path, heuristic, pheromone = infer_instance(model, pyg_data, distances, n_ants, t_aco_diff, k_sparse)
        sum_results += results
        last_heuristic = heuristic
        last_pheromone = pheromone
    end = time.time()

    return sum_results / len(dataset), best_path, end - start, last_heuristic, last_pheromone

def main(n_node, model_file, k_sparse = None, n_ants=20, t_aco = None):
    k_sparse =k_sparse or n_node
    t_aco = None or list(range(1,10))
    test_list = load_test_dataset(n_node, k_sparse, device, start_node = None )
    print("problem scale:", n_node)
    print("checkpoint:", model_file)
    print("number of instances:", len(test_list))
    print("device:", 'cpu' if device == 'cpu' else device+"+cpu" )

    net_tsp = Net().to(device)
    net_tsp.load_state_dict(torch.load(model_file, map_location=device))
    # avg_aco_best, duration = test(test_list, net_tsp, n_ants, t_aco, k_sparse)
    avg_aco_best, best_path, duration, last_heuristic, last_pheromone = test(test_list, net_tsp, n_ants, t_aco, k_sparse)
    print('total duration: ', duration)
    for i, t in enumerate(t_aco):
        print("T={}, average cost is {}.".format(t, avg_aco_best[i]))
    torch.save(best_path, '../plot/exam_path1001.pt')
    torch.save(last_heuristic, '../plot/exam_heu.pt')
    torch.save(last_pheromone, '../plot/exam_phe.pt')


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("-nodes", type=int, default=100, help="Problem scale")
    parser.add_argument("-m", "--model", type=str, default=None, help=f"Path to checkpoint file, default to '../pretrained/tsp_nls/tsp100-best.pt'")
    opt = parser.parse_args()
    n_nodes = opt.nodes

    filepath = opt.model or f'../pretrained/tsp_nls/tsp100-best.pt'
    if not os.path.isfile(filepath):
        print(f"Checkpoint file '{filepath}' not found!")
        exit(5)
    
    main(n_nodes, filepath)
