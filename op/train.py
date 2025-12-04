import time
import torch
from torch.distributions import Categorical, kl
from d2l.torch import Animator

from transformer_encoder import Net
from aco import ACO
from utils import *

lr = 3e-4
EPS = 1e-10
device = 'cuda:0'

max_len = {
    50: 3,
    100: 4,
    200: 5,
    300: 6
}


# %%
def train_instance(model, optimizer, n, k_sparse, n_ants):
    model.train()
    coor = torch.rand(size=(n, 2), device=device)
    pyg_data, distances, prizes = gen_pyg_data(coor, k_sparse)
    phe_mat, heu_mat = model(pyg_data, distances)
    aco = ACO(distances, prizes, max_len[n], n_ants, heuristic=heu_mat, pheromone=phe_mat, device=device)
    objs, log_probs = aco.sample()
    baseline = objs.mean()
    reinforce_loss = torch.sum((baseline - objs) * log_probs.sum(dim=0)) / aco.n_ants
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()


def infer_instance(model, instance, k_sparse, n_ants):
    model.eval()
    pyg_data, distances, prizes = instance
    phe_mat, heu_mat = model(pyg_data, distances)
    aco = ACO(distances, prizes, max_len[len(prizes)], n_ants, heuristic=heu_mat, pheromone=phe_mat, device=device)
    objs, log_probs_list = aco.sample()
    baseline = objs.mean()
    best_sample_obj = objs.max()
    return baseline.item(), best_sample_obj.item()


# %%
def train_epoch(n, n_ants, k_sparse, steps_per_epoch, net, optimizer):
    for _ in range(steps_per_epoch):
        train_instance(net, optimizer, n, k_sparse, n_ants)


@torch.no_grad()
def validation(n_ants, k_sparse, epoch, net, val_dataset, animator=None):
    sum_bl, sum_sample_best = 0, 0
    for instance in val_dataset:
        bl, sample_best = infer_instance(net, instance, k_sparse, n_ants)
        sum_bl += bl
        sum_sample_best += sample_best

    n_val = len(val_dataset)
    avg_bl, avg_sample_best = sum_bl / n_val, sum_sample_best / n_val
    if animator:
        animator.add(epoch + 1, (avg_bl, avg_sample_best))

    return avg_bl, avg_sample_best


# %%
def train(problem_size, k_sparse, n_ants, steps_per_epoch, epochs):
    net = Net().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    val_list = load_val_dataset(problem_size, k_sparse, device)
    animator = Animator(xlabel='epoch', xlim=[0, epochs],
                        legend=["Avg. sample obj.", "Best sample obj."])

    avg_bl, avg_best = validation(n_ants, k_sparse, -1, net, val_list, animator)
    val_results = [(avg_bl, avg_best)]

    all_time_best = 0

    sum_time = 0
    for epoch in range(0, epochs):
        start = time.time()
        train_epoch(problem_size, n_ants, k_sparse, steps_per_epoch, net, optimizer)
        sum_time += time.time() - start
        avg_bl, avg_sample_best = validation(n_ants, k_sparse, epoch, net, val_list, animator)
        val_results.append((avg_bl, avg_sample_best))
        print(f'epoch {epoch + 1}:', avg_bl, avg_sample_best)
        if avg_sample_best > all_time_best:
            all_time_best = avg_sample_best
            torch.save(net.state_dict(), f'../pretrained/op/op{problem_size}.pt')

    print('total training duration:', sum_time)

    for epoch in range(-1, epochs):
        print(f'epoch {epoch}:', val_results[epoch + 1])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", metavar='N', type=int, default=50, help="Problem scale")
    parser.add_argument("-l", "--lr", metavar='η', type=float, default=1e-4, help="Learning rate")
    parser.add_argument("-d", "--device", type=str,
                        default=("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help="The device to train NNs")
    parser.add_argument("-p", "--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("-k", "--k_sparse", type=int, default=20, help="Path to pretrained model")
    parser.add_argument("-a", "--ants", type=int, default=30, help="Number of ants (in ACO algorithm)")
    parser.add_argument("-b", "--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("-s", "--steps", type=int, default=20, help="Steps per epoch")
    parser.add_argument("-e", "--epochs", type=int, default=7, help="Epochs to run")
    parser.add_argument("-t", "--test_size", type=int, default=10, help="Number of instances for testing")
    parser.add_argument("-o", "--output", type=str, default="../pretrained/cvrp",
                        help="The directory to store checkpoints")
    opt = parser.parse_args()

    lr = opt.lr
    device = opt.device
    n_node = opt.nodes

    train(
        opt.nodes,
        opt.k_sparse,
        opt.ants,
        opt.steps,
        opt.epochs,
    )