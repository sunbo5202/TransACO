import torch
from torch import Tensor
import numpy as np
from torch_geometric.data import Data

def gen_instance(n, m, device):
    '''
    Generate *well-stated* MKP instances
    Args:
        n: # of knapsacks
        m: # of constraints, a.k.a., the problem dimensionality 
    '''
    prize = torch.rand(size=(n,), device=device)
    weight_matrix = torch.rand(size=(n, m), device=device)
    max_weight, _ = torch.max(weight_matrix, dim=0)
    sum_weight = torch.sum(weight_matrix, dim=0)
    constraints = []
    for idx in range(m):
        constraint = np.random.uniform(low=max_weight[idx].item(), high=sum_weight[idx].item())
        constraints.append(constraint)
    constraints = torch.tensor(constraints, device=device)
    # after norm, constraints are all n//2
    weight_matrix = weight_matrix * (n//2) / constraints.unsqueeze(0) 
    return prize, weight_matrix # (n, ), (n, m)

def gen_pyg_data(prize, weight_matrix):
    device = prize.device
    n = prize.size(0)
    x = weight_matrix
    nodes = torch.arange(n, device=device)
    u = nodes.repeat(n)
    v = torch.repeat_interleave(nodes, n)
    edge_attr = prize.repeat(n).unsqueeze(-1)
    pyg_data = Data(x=x, edge_index=torch.stack((u, v)), edge_attr=edge_attr)
    return pyg_data

def load_val_dataset(problem_size, device):
    """
    Load validation dataset.
    """
    import os
    
    # Try multiple possible paths
    possible_paths = [
        f'../data/mkp/valDataset-{problem_size}.pt',
        f'./data/mkp/valDataset-{problem_size}.pt',
    ]
    
    dataset = None
    for path in possible_paths:
        if os.path.isfile(path):
            dataset = torch.load(path, map_location=device)
            break
    
    if dataset is None:
        raise FileNotFoundError(f"Validation dataset not found. Tried: {possible_paths}")
    
    val_list = []
    for i in range(len(dataset)):
        val_list.append((dataset[i, :, 0], dataset[i, :, 1:]))
    return val_list

def load_test_dataset(problem_size, device):
    """
    Load test dataset. If file doesn't exist, generate 10 instances automatically.
    """
    import os
    
    test_file = f'./data/mkp/testDataset-{problem_size}.pt'
    if not os.path.isfile(test_file):
        # Create directory if it doesn't exist
        os.makedirs('./data/mkp', exist_ok=True)
        # Generate test dataset with 10 instances
        M = 5
        torch.manual_seed(123456)
        testDataset = []
        for _ in range(10):
            prize, weight = gen_instance(problem_size, M, device)
            testDataset.append(torch.cat((prize.unsqueeze(1), weight), dim=1))
        testDataset = torch.stack(testDataset)
        torch.save(testDataset, test_file)
        print(f"Generated test dataset with 10 instances: {test_file}")
    
    dataset = torch.load(test_file, map_location=device)
    val_list = []
    for i in range(len(dataset)):
        val_list.append((dataset[i, :, 0], dataset[i, :, 1:]))
    return val_list

if __name__ == '__main__':
    # generate val and test dataset
    import os
    import pathlib
    
    # Ensure directories exist
    pathlib.Path('./data/mkp').mkdir(parents=True, exist_ok=True)
    pathlib.Path('../data/mkp').mkdir(parents=True, exist_ok=True)
    
    M = 5
    
    # Generate validation dataset
    torch.manual_seed(12345)
    for problem_size in [100, 300, 500]:
        val_file = f'../data/mkp/valDataset-{problem_size}.pt'
        if not os.path.isfile(val_file):
            valDataset = []
            for _ in range(100):
                prize, weight = gen_instance(problem_size, M, 'cpu')
                valDataset.append(torch.cat((prize.unsqueeze(1), weight), dim=1))
            valDataset = torch.stack(valDataset)
            torch.save(valDataset, val_file)
            print(f"Generated validation dataset: {val_file}")
    
    # Generate test dataset (10 instances)
    torch.manual_seed(123456)
    for problem_size in [100, 300, 500]:
        # Save to both ./data/mkp and ../data/mkp for compatibility
        test_files = [
            f'./data/mkp/testDataset-{problem_size}.pt',
            f'../data/mkp/testDataset-{problem_size}.pt'
        ]
        for test_file in test_files:
            if not os.path.isfile(test_file):
                os.makedirs(os.path.dirname(test_file), exist_ok=True)
                testDataset = []
                for _ in range(10):  # Generate 10 test instances
                    prize, weight = gen_instance(problem_size, M, 'cpu')
                    testDataset.append(torch.cat((prize.unsqueeze(1), weight), dim=1))
                testDataset = torch.stack(testDataset)
                torch.save(testDataset, test_file)
                print(f"Generated test dataset with 10 instances: {test_file}")
    
    