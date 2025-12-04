import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch

# ===== PEG (Population Evolution Graph) Initialization =====
ACO_PEG = nx.DiGraph()  # Global PEG graph for ACO

def log_aco_to_peg(paths, objs, iteration, instance_id=0):
    """
    Log ACO paths to PEG graph for MKP (maximization problem).
    
    Args:
        paths: torch tensor with shape (max_horizon, n_ants) for MKP
        objs: torch tensor with shape (n_ants,) - reward values (higher is better)
        iteration: current iteration number
        instance_id: instance identifier for multiple problem instances
    """
    global ACO_PEG
    
    # Validate inputs
    if paths is None or objs is None:
        print(f"ERROR: log_aco_to_peg called with None inputs (iteration={iteration}, instance_id={instance_id})")
        return None
    
    # Convert to proper format if needed
    if not isinstance(paths, torch.Tensor):
        paths = torch.tensor(paths)
    if not isinstance(objs, torch.Tensor):
        objs = torch.tensor(objs)
    
    # Ensure paths is 2D and objs is 1D
    if len(paths.shape) != 2:
        print(f"ERROR: paths.shape={paths.shape}, expected 2D tensor (iteration={iteration}, instance_id={instance_id})")
        return None
    if len(objs.shape) != 1:
        if objs.dim() == 0:
            print(f"ERROR: objs is scalar, expected 1D tensor (iteration={iteration}, instance_id={instance_id})")
            return None
        objs = objs.squeeze()
        if len(objs.shape) != 1:
            print(f"ERROR: objs.shape={objs.shape} after squeeze, expected 1D tensor (iteration={iteration}, instance_id={instance_id})")
            return None
    
    # For MKP, paths shape should be (max_horizon, n_ants)
    # But after transposition in run_aco_with_peg, it might be (n_ants, max_horizon)
    # We need to ensure paths.shape[1] == objs.shape[0] (n_ants)
    n_ants = objs.shape[0]
    original_shape = paths.shape
    
    # Check which dimension corresponds to n_ants
    if paths.shape[1] == n_ants:
        # Paths is already (max_horizon, n_ants), which is correct
        pass
    elif paths.shape[0] == n_ants:
        # Paths is (n_ants, max_horizon), need to transpose
        paths = paths.T
        print(f"Debug: Transposed paths from {original_shape} to {paths.shape} (iteration={iteration}, instance_id={instance_id})")
    else:
        # Neither dimension matches n_ants, this is an error
        print(f"ERROR: Cannot determine paths format. paths.shape={paths.shape}, n_ants={n_ants} (iteration={iteration}, instance_id={instance_id})")
        return None
    
    # Final validation
    if paths.shape[1] != objs.shape[0]:
        print(f"ERROR: Mismatch between paths.shape[1]={paths.shape[1]} and objs.shape[0]={objs.shape[0]} (iteration={iteration}, instance_id={instance_id})")
        return None
    
    # Find best path in current iteration (MKP maximizes rewards, so use argmax)
    best_idx = objs.argmax().item()
    best_obj = objs[best_idx].item()
    
    # Add all paths in current iteration as nodes
    nodes_added = 0
    current_paths_dict = {}  # Store current paths for pheromone-based connection
    
    for ant_id in range(paths.shape[1]):
        node_id = f"inst_{instance_id}_iter_{iteration}_ant_{ant_id}"
        is_best = (ant_id == best_idx)
        path_list = paths[:, ant_id].cpu().numpy().tolist()
        
        # For MKP, reward value (higher is better) - store as 'cost' for compatibility (but it's actually reward)
        
        # Add node with attributes
        ACO_PEG.add_node(
            node_id,
            iteration=iteration,
            instance_id=instance_id,
            ant_id=ant_id,
            cost=objs[ant_id].item(),  # Actually reward value (higher is better)
            is_best=is_best,
            path=path_list,
        )
        current_paths_dict[ant_id] = {
            'node_id': node_id,
            'path': path_list,
            'reward': objs[ant_id].item(),
            'is_best': is_best
        }
        nodes_added += 1
    
    # Connect paths: link each ant to its previous iteration (simple chain connection)
    if iteration > 0:
        # Get all previous iteration nodes for this instance
        prev_nodes = [n for n in ACO_PEG.nodes() 
                     if ACO_PEG.nodes[n].get('instance_id') == instance_id 
                     and ACO_PEG.nodes[n].get('iteration') == iteration - 1]
        
        # For each current ant, connect to the same ant_id in previous iteration
        for curr_ant_id, curr_data in current_paths_dict.items():
            curr_node_id = curr_data['node_id']
            
            # Find the node with same ant_id in previous iteration
            for prev_node in prev_nodes:
                prev_ant_id = ACO_PEG.nodes[prev_node].get('ant_id', -1)
                if prev_ant_id == curr_ant_id:
                    # Connect same ant across iterations (simple chain)
                    if not ACO_PEG.has_edge(prev_node, curr_node_id):
                        ACO_PEG.add_edge(
                            prev_node, 
                            curr_node_id,
                            weight=1.0
                        )
                    break  # Found the matching ant, move to next current ant
    
    # Debug: print when nodes are added
    if nodes_added > 0:
        print(f"Debug: Added {nodes_added} nodes to PEG (instance={instance_id}, iteration={iteration}, total_nodes={ACO_PEG.number_of_nodes()})")
    
    return f"inst_{instance_id}_iter_{iteration}_ant_{best_idx}"

def visualize_aco_peg(save_path=None, instance_id=None, layout_type='hierarchical'):
    """
    Visualize ACO PEG graph for MKP (maximization problem).
    
    Args:
        save_path: Path to save the figure (optional)
        instance_id: Filter by instance_id if provided
        layout_type: 'hierarchical' for iteration-based layout or 'spring' for force-directed layout
    """
    global ACO_PEG
    
    if ACO_PEG.number_of_nodes() == 0:
        print("PEG graph is empty. No visualization generated.")
        return
    
    # Filter nodes by instance_id if provided
    if instance_id is not None:
        nodes_to_plot = [n for n in ACO_PEG.nodes() 
                        if ACO_PEG.nodes[n].get('instance_id') == instance_id]
        subgraph = ACO_PEG.subgraph(nodes_to_plot)
    else:
        subgraph = ACO_PEG
    
    if subgraph.number_of_nodes() == 0:
        print(f"No nodes found for instance_id={instance_id}")
        return
    
    # Get iterations and ants
    iterations = sorted(set(subgraph.nodes[n].get('iteration', 0) 
                           for n in subgraph.nodes()))
    max_ant_id = max(subgraph.nodes[n].get('ant_id', 0) for n in subgraph.nodes())
    n_ants = max_ant_id + 1
    
    # Create layout based on type
    if layout_type == 'hierarchical':
        # Hierarchical layout: iterations on x-axis, ants on y-axis
        pos = {}
        iteration_positions = {}
        
        # Calculate spacing
        iter_spacing = 2.0  # Horizontal spacing between iterations
        ant_spacing = 1.0   # Vertical spacing between ants
        
        for i, iter_num in enumerate(iterations):
            iteration_positions[iter_num] = i * iter_spacing
            
            # Group nodes by ant_id within this iteration
            for ant_id in range(n_ants):
                nodes_for_ant = [n for n in subgraph.nodes() 
                                if subgraph.nodes[n].get('iteration') == iter_num 
                                and subgraph.nodes[n].get('ant_id') == ant_id]
                if nodes_for_ant:
                    # Place at ant_id position vertically
                    y_pos = (ant_id - n_ants / 2) * ant_spacing
                    for node in nodes_for_ant:
                        pos[node] = (i * iter_spacing, y_pos)
    else:
        # Spring layout (force-directed) like EvoMapX
        pos = nx.spring_layout(subgraph, k=1.5, iterations=50, seed=42)
    
    # Prepare node colors - use reward-based coloring (higher reward = better color for MKP)
    node_colors = []
    node_sizes = []
    rewards = [subgraph.nodes[n].get('cost', 0) for n in subgraph.nodes()]  # 'cost' is actually reward
    
    if rewards:
        min_reward = min(rewards)
        max_reward = max(rewards)
        reward_range = max_reward - min_reward if max_reward > min_reward else 1
        
        for node in subgraph.nodes():
            reward = subgraph.nodes[node].get('cost', 0)
            # Normalize reward to [0, 1] for colormap (higher reward = brighter color)
            normalized_reward = (reward - min_reward) / reward_range if reward_range > 0 else 0
            node_colors.append(normalized_reward)
            
            # Make best paths larger and use different style
            if subgraph.nodes[node].get('is_best', False):
                node_sizes.append(150)
            else:
                node_sizes.append(50)
    
    # Create figure with better settings
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Draw edges (if any exist)
    edge_weights = []
    edge_colors = []
    for edge in subgraph.edges():
        edge_data = subgraph.edges[edge]
        weight = edge_data.get('weight', 0.5)  # Default weight
        
        # Scale weight for visualization
        edge_weights.append(weight if weight > 0 else 0.5)
        edge_colors.append('gray')  # Simple gray color for all edges
    
    if not edge_weights:
        edge_weights = [0.5] * len(subgraph.edges())
        edge_colors = ['gray'] * len(subgraph.edges())
    
    # Draw edges first (networkx draws edges before nodes by default)
    nx.draw_networkx_edges(
        subgraph,
        pos,
        edge_color=edge_colors,
        alpha=0.5,
        arrows=True,
        arrowsize=8,
        arrowstyle='->',
        width=edge_weights,
        ax=ax
    )
    
    # Draw nodes on top of edges (networkx draws nodes after edges by default)
    nodes_list = list(subgraph.nodes())
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=nodes_list,
        node_color=node_colors,
        node_size=node_sizes,
        cmap='viridis',  # Higher reward = brighter color (not reversed for MKP)
        alpha=0.8,
        edgecolors='black',
        linewidths=1.5,
        ax=ax
    )
    
    # Add labels for best paths only (to reduce clutter)
    best_nodes = [n for n in subgraph.nodes() 
                 if subgraph.nodes[n].get('is_best', False)]
    if best_nodes:
        best_labels = {n: f"T{subgraph.nodes[n].get('iteration', 0)}" 
                      for n in best_nodes}
        nx.draw_networkx_labels(
            subgraph, pos, best_labels, 
            font_size=10, font_color='red', font_weight='bold',
            ax=ax
        )
    
    # Add colorbar
    if rewards:
        sm = plt.cm.ScalarMappable(
            cmap='viridis', 
            norm=plt.Normalize(vmin=min_reward, vmax=max_reward)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Reward Value', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    
    # Set title and labels
    title = f"ACO Population Evolution Graph (PEG) - MKP"
    if instance_id is not None:
        title += f" - Instance {instance_id}"
    title += f"\n({layout_type.capitalize()} Layout)"
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    if layout_type == 'hierarchical':
        plt.xlabel("Iteration", fontsize=12, fontweight='bold')
        plt.ylabel("Ant Index", fontsize=12, fontweight='bold')
        # Add iteration labels on x-axis
        ax.set_xticks([i * iter_spacing for i in range(len(iterations))])
        ax.set_xticklabels([f"I{iter_num}" for iter_num in iterations])
        plt.grid(True, alpha=0.3, linestyle='--')
    else:
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        # Save PNG format
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"PEG visualization saved to {save_path}")
        
        # Also save PDF format
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"PEG visualization saved to {pdf_path} (PDF)")
    
    plt.close()

def visualize_aco_peg_simple(save_path=None, instance_id=None):
    """
    Simplified PEG visualization inspired by EvoMapX style - using spring layout.
    This provides a cleaner, more compact view of the evolution structure.
    
    Args:
        save_path: Path to save the figure (optional)
        instance_id: Filter by instance_id if provided
    """
    global ACO_PEG
    
    if ACO_PEG.number_of_nodes() == 0:
        print("PEG graph is empty. No visualization generated.")
        return
    
    # Filter nodes by instance_id if provided
    if instance_id is not None:
        nodes_to_plot = [n for n in ACO_PEG.nodes() 
                        if ACO_PEG.nodes[n].get('instance_id') == instance_id]
        subgraph = ACO_PEG.subgraph(nodes_to_plot)
    else:
        subgraph = ACO_PEG
    
    if subgraph.number_of_nodes() == 0:
        print(f"No nodes found for instance_id={instance_id}")
        return
    
    # Use spring layout like EvoMapX
    pos = nx.spring_layout(subgraph, k=1.5, iterations=50, seed=42)
    
    # Color nodes by ant_id (each ant gets a unique color in its chain)
    # Group by ant_id to give each ant a distinct color scheme
    unique_ant_ids = sorted(set(subgraph.nodes[n].get('ant_id', 0) 
                               for n in subgraph.nodes()))
    
    # Use colormap for ant_id coloring
    n_ants = len(unique_ant_ids)
    ant_colors = plt.cm.tab20(np.linspace(0, 1, min(n_ants, 20)))
    if n_ants > 20:
        ant_colors = plt.cm.Set3(np.linspace(0, 1, n_ants))
    
    node_colors = []
    node_sizes = []
    
    for node in subgraph.nodes():
        ant_id = subgraph.nodes[node].get('ant_id', 0)
        ant_idx = unique_ant_ids.index(ant_id) if ant_id in unique_ant_ids else 0
        node_colors.append(ant_colors[ant_idx % len(ant_colors)])
        
        # Make best paths larger
        if subgraph.nodes[node].get('is_best', False):
            node_sizes.append(200)
        else:
            node_sizes.append(80)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Draw edges first (networkx draws edges before nodes by default)
    nx.draw_networkx_edges(
        subgraph,
        pos,
        edge_color='gray',
        alpha=0.3,
        arrows=True,
        arrowsize=10,
        width=0.5
    )
    
    # Draw nodes on top of edges (networkx draws nodes after edges by default)
    nodes_list = list(subgraph.nodes())
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=nodes_list,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.7,
        edgecolors='black',
        linewidths=1.0
    )
    
    # Optional: label only best paths to reduce clutter
    best_nodes = [n for n in subgraph.nodes() 
                 if subgraph.nodes[n].get('is_best', False)]
    if len(best_nodes) <= 20:  # Only label if not too many
        best_labels = {n: f"T{subgraph.nodes[n].get('iteration', 0)}" 
                      for n in best_nodes}
        nx.draw_networkx_labels(
            subgraph, pos, best_labels,
            font_size=8, font_color='red', font_weight='bold'
        )
    
    title = "ACO Population Evolution Graph (PEG) - MKP"
    if instance_id is not None:
        title += f" - Instance {instance_id}"
    title += " (Spring Layout)"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Add legend for ant chains (if not too many)
    if n_ants <= 10:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=ant_colors[i % len(ant_colors)], 
                  label=f'Ant {unique_ant_ids[i]}')
            for i in range(min(n_ants, 10))
        ]
        plt.legend(handles=legend_elements, loc='upper right', 
                  title="Ant ID", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        # Modify save path to add '_simple' suffix
        if save_path.endswith('.png'):
            simple_save_path = save_path.replace('.png', '_simple.png')
        else:
            simple_save_path = save_path + '_simple'
        
        # Save PNG format
        plt.savefig(simple_save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Simple PEG visualization saved to {simple_save_path}")
        
        # Also save PDF format
        simple_pdf_path = simple_save_path.replace('.png', '.pdf')
        plt.savefig(simple_pdf_path, bbox_inches='tight', facecolor='white')
        print(f"Simple PEG visualization saved to {simple_pdf_path} (PDF)")
    
    plt.close()

def reset_peg():
    """Reset the global PEG graph."""
    global ACO_PEG
    ACO_PEG = nx.DiGraph()

def analyze_aco_peg_metrics(instance_id=None):
    """
    Analyze ACO PEG graph to provide quantitative insights into search behavior.
    For MKP, this analyzes reward maximization.
    
    Args:
        instance_id: Filter by instance_id if provided
    
    Returns:
        Dictionary containing quantitative metrics and insights
    """
    global ACO_PEG
    
    if ACO_PEG.number_of_nodes() == 0:
        return {"error": "PEG graph is empty"}
    
    # Filter by instance if provided
    if instance_id is not None:
        nodes_to_analyze = [n for n in ACO_PEG.nodes() 
                           if ACO_PEG.nodes[n].get('instance_id') == instance_id]
        subgraph = ACO_PEG.subgraph(nodes_to_analyze)
    else:
        subgraph = ACO_PEG
    
    if subgraph.number_of_nodes() == 0:
        return {"error": f"No nodes found for instance_id={instance_id}"}
    
    metrics = {}
    
    # 1. Basic graph structure metrics
    metrics['nodes'] = subgraph.number_of_nodes()
    metrics['edges'] = subgraph.number_of_edges()
    n = metrics['nodes']
    metrics['density'] = metrics['edges'] / (n * (n - 1)) if n > 1 else 0
    
    # 2. Centrality analysis (identify influential paths)
    in_degrees = dict(subgraph.in_degree())
    out_degrees = dict(subgraph.out_degree())
    metrics['avg_in_degree'] = np.mean(list(in_degrees.values())) if in_degrees else 0
    metrics['max_in_degree'] = max(in_degrees.values()) if in_degrees else 0
    metrics['avg_out_degree'] = np.mean(list(out_degrees.values())) if out_degrees else 0
    
    # Find most influential nodes (highest in-degree)
    if in_degrees:
        max_in_degree = max(in_degrees.values())
        influential_nodes = [n for n, d in in_degrees.items() if d == max_in_degree]
        metrics['most_influential_count'] = len(influential_nodes)
        metrics['most_influential_nodes'] = influential_nodes[:5]  # Top 5
    
    # 3. Reward evolution analysis (for MKP, we maximize rewards)
    iterations = sorted(set(subgraph.nodes[n].get('iteration', 0) 
                           for n in subgraph.nodes()))
    metrics['total_iterations'] = len(iterations)
    
    best_rewards_per_iter = []  # For MKP, higher is better
    avg_rewards_per_iter = []
    for iter_num in iterations:
        iter_nodes = [n for n in subgraph.nodes() 
                     if subgraph.nodes[n].get('iteration') == iter_num]
        if iter_nodes:
            iter_rewards = [subgraph.nodes[n].get('cost', 0) for n in iter_nodes]  # 'cost' is actually reward
            best_rewards_per_iter.append(max(iter_rewards))  # max for MKP
            avg_rewards_per_iter.append(np.mean(iter_rewards))
    
    if len(best_rewards_per_iter) > 1:
        metrics['initial_best_reward'] = best_rewards_per_iter[0]
        metrics['final_best_reward'] = best_rewards_per_iter[-1]
        metrics['total_improvement'] = best_rewards_per_iter[-1] - best_rewards_per_iter[0]  # final - initial for MKP
        metrics['relative_improvement'] = (metrics['total_improvement'] / best_rewards_per_iter[0]) * 100 if best_rewards_per_iter[0] > 0 else 0
        
        # Convergence rate (how fast it improves)
        improvements = [best_rewards_per_iter[i+1] - best_rewards_per_iter[i] 
                       for i in range(len(best_rewards_per_iter)-1)]
        metrics['avg_iteration_improvement'] = np.mean(improvements) if improvements else 0
        metrics['convergence_rate'] = metrics['avg_iteration_improvement'] / metrics['total_improvement'] if metrics['total_improvement'] > 0 else 0
    
    # 4. Influence analysis (which paths influenced more descendants)
    influence_scores = {}
    for node in subgraph.nodes():
        try:
            descendants = list(nx.descendants(subgraph, node))
            influence_scores[node] = len(descendants)
        except:
            influence_scores[node] = 0
    
    if influence_scores:
        metrics['max_influence'] = max(influence_scores.values())
        metrics['avg_influence'] = np.mean(list(influence_scores.values()))
        # Find most influential paths
        max_influence = max(influence_scores.values())
        most_influential = [n for n, score in influence_scores.items() if score == max_influence]
        metrics['most_influential_paths'] = most_influential[:5]
    
    # 5. Population diversity analysis
    if len(best_rewards_per_iter) > 1:
        reward_variance_per_iter = []
        for iter_num in iterations:
            iter_nodes = [n for n in subgraph.nodes() 
                         if subgraph.nodes[n].get('iteration') == iter_num]
            if iter_nodes:
                iter_rewards = [subgraph.nodes[n].get('cost', 0) for n in iter_nodes]
                reward_variance_per_iter.append(np.var(iter_rewards))
        
        metrics['avg_reward_variance'] = np.mean(reward_variance_per_iter) if reward_variance_per_iter else 0
        metrics['diversity_trend'] = 'increasing' if reward_variance_per_iter[-1] > reward_variance_per_iter[0] else 'decreasing'
    
    # 6. Best path persistence (how often best path changes)
    best_path_changes = 0
    prev_best_reward = None
    for iter_num in iterations:
        iter_best_nodes = [n for n in subgraph.nodes() 
                          if subgraph.nodes[n].get('iteration') == iter_num
                          and subgraph.nodes[n].get('is_best', False)]
        if iter_best_nodes:
            current_best_reward = subgraph.nodes[iter_best_nodes[0]].get('cost', 0)
            if prev_best_reward is not None and current_best_reward != prev_best_reward:
                best_path_changes += 1
            prev_best_reward = current_best_reward
    
    metrics['best_path_changes'] = best_path_changes
    metrics['stability'] = 1 - (best_path_changes / len(iterations)) if iterations else 0
    
    return metrics

def explain_aco_peg_behavior(instance_id=None):
    """
    Generate textual explanation of ACO search behavior based on PEG analysis.
    For MKP (maximization problem).
    
    Args:
        instance_id: Filter by instance_id if provided
    
    Returns:
        String containing detailed explanation
    """
    metrics = analyze_aco_peg_metrics(instance_id)
    
    if 'error' in metrics:
        return f"Error: {metrics['error']}"
    
    explanation = "=" * 80 + "\n"
    explanation += "ACO SEARCH BEHAVIOR ANALYSIS (Based on Population Evolution Graph) - MKP\n"
    explanation += "=" * 80 + "\n\n"
    
    # Graph structure
    explanation += "1. POPULATION STRUCTURE:\n"
    explanation += f"   - Total individuals explored: {metrics['nodes']}\n"
    explanation += f"   - Evolutionary relationships: {metrics['edges']}\n"
    explanation += f"   - Graph density: {metrics['density']:.6f}\n"
    if metrics['density'] > 0.1:
        explanation += "   → High connectivity indicates strong information sharing\n"
    else:
        explanation += "   → Sparse connectivity suggests independent search paths\n"
    explanation += "\n"
    
    # Convergence analysis (for maximization)
    if 'total_improvement' in metrics:
        explanation += "2. CONVERGENCE BEHAVIOR:\n"
        explanation += f"   - Initial best reward: {metrics['initial_best_reward']:.2f}\n"
        explanation += f"   - Final best reward: {metrics['final_best_reward']:.2f}\n"
        explanation += f"   - Total improvement: {metrics['total_improvement']:.2f} ({metrics['relative_improvement']:.2f}%)\n"
        explanation += f"   - Average improvement per iteration: {metrics['avg_iteration_improvement']:.2f}\n"
        if metrics['convergence_rate'] > 0.5:
            explanation += "   → Fast convergence: significant improvements in early iterations\n"
        else:
            explanation += "   → Gradual convergence: steady improvements throughout\n"
        explanation += "\n"
    
    # Influence analysis
    explanation += "3. INFLUENCE ANALYSIS:\n"
    explanation += f"   - Maximum influence (descendants): {metrics['max_influence']}\n"
    explanation += f"   - Average influence: {metrics['avg_influence']:.2f}\n"
    if metrics['max_influence'] > metrics['avg_influence'] * 2:
        explanation += "   → Some paths have significantly more influence than others\n"
    else:
        explanation += "   → Relatively uniform influence distribution\n"
    explanation += "\n"
    
    # Stability analysis
    if 'stability' in metrics:
        explanation += "4. SEARCH STABILITY:\n"
        explanation += f"   - Best path changes: {metrics['best_path_changes']} times\n"
        explanation += f"   - Stability score: {metrics['stability']:.2f}\n"
        if metrics['stability'] > 0.7:
            explanation += "   → Stable search: best solution persists across iterations\n"
        else:
            explanation += "   → Dynamic search: frequent changes in best solution\n"
        explanation += "\n"
    
    # Diversity analysis
    if 'avg_reward_variance' in metrics:
        explanation += "5. POPULATION DIVERSITY:\n"
        explanation += f"   - Average reward variance: {metrics['avg_reward_variance']:.2f}\n"
        explanation += f"   - Diversity trend: {metrics['diversity_trend']}\n"
        if metrics['diversity_trend'] == 'increasing':
            explanation += "   → Increasing diversity: population explores wider solution space\n"
        else:
            explanation += "   → Decreasing diversity: population converging to similar solutions\n"
        explanation += "\n"
    
    explanation += "=" * 80 + "\n"
    
    return explanation

def run_aco_with_peg(aco, n_iterations, inference=False, instance_id=0, start_iteration=0):
    """
    Wrapper function to run MKP ACO and log to PEG graph.
    This function replicates aco.run() but adds PEG logging.
    
    Args:
        aco: ACO instance (for MKP)
        n_iterations: Number of iterations to run
        inference: Whether in inference mode (not used for MKP)
        instance_id: Instance identifier for PEG
        start_iteration: Starting iteration number for PEG logging (for cumulative tracking)
    
    Returns:
        Same as aco.run() - (alltime_best_obj, alltime_best_sol)
    """
    for i in range(n_iterations):
        sols = aco.gen_sol(require_prob=False)  # Returns (max_horizon, n_ants)
        objs = aco.gen_sol_obj(sols)  # Returns (n_ants,)
        sols = sols.T  # Transpose to (n_ants, max_horizon) for consistency with run()
        
        best_obj, best_idx = objs.max(dim=0)  # MKP maximizes rewards
        if best_obj > aco.alltime_best_obj:
            aco.alltime_best_obj = best_obj
            aco.alltime_best_sol = sols[best_idx]
        aco.update_pheronome(sols, objs, best_obj.item(), best_idx.item())
        
        # Log to PEG AFTER pheromone update
        # sols is (n_ants, max_horizon), transpose back to (max_horizon, n_ants) for PEG
        log_aco_to_peg(
            sols.T,  # Back to (max_horizon, n_ants)
            objs,    # (n_ants,) - reward values
            iteration=start_iteration + i, 
            instance_id=instance_id
        )
    
    return aco.alltime_best_obj, aco.alltime_best_sol

