import os
import time
import pickle
import random
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm
from collections import deque



class GraphData:
    def __init__(self, adj_matrix, num_nodes):
        self.adj_matrix = adj_matrix 
        self.num_nodes = num_nodes
        self.rev_adj_list = self._build_reverse_adj_list()

    def _build_reverse_adj_list(self):
        rev_adj = self.adj_matrix.transpose().tocoo()
        adj_list = [[] for _ in range(self.num_nodes)]
        for u, v, p in zip(rev_adj.row, rev_adj.col, rev_adj.data):
            adj_list[u].append((v, p))
        return adj_list



class Sampler_IC:
    def __init__(self, graph_data):
        self.graph = graph_data

    def infer_generate(self, num_samples):
        rr_sets = []
        roots = np.random.randint(0, self.graph.num_nodes, num_samples)
        for root in roots:
            rr_set = {root}
            q = [root]
            while q:
                u = q.pop(0)
                if u >= len(self.graph.rev_adj_list): continue
                neighbors = self.graph.rev_adj_list[u]
                for v, prob in neighbors:
                    if v not in rr_set:
                        if random.random() <= prob:
                            rr_set.add(v)
                            q.append(v)
            rr_sets.append(rr_set)
        return rr_sets, roots
    


class Sampler_LT:
    def __init__(self, graph_data):
        self.graph = graph_data

    def infer_generate(self, num_samples):
        rr_sets = []
        roots = np.random.randint(0, self.graph.num_nodes, num_samples)
        for root in roots:
            rr_set = {root}
            curr = root
            while True:
                if curr >= len(self.graph.rev_adj_list): 
                    break
                neighbors_info = self.graph.rev_adj_list[curr]
                if not neighbors_info:
                    break
                neighbors, weights = zip(*neighbors_info)
                next_node = random.choices(neighbors, weights=weights, k=1)[0]
                if next_node in rr_set:
                    break 
                rr_set.add(next_node)
                curr = next_node

            rr_sets.append(rr_set)
        return rr_sets, roots



class GraphDataForSIS:
    def __init__(self, adj_matrix, num_nodes):
        self.adj_matrix = adj_matrix
        self.num_nodes = num_nodes
        self.rev_adj_list = self._build_reverse_adj_list()
    def _build_reverse_adj_list(self):
        rev_adj = self.adj_matrix.transpose().tocoo()
        adj_list = [[] for _ in range(self.num_nodes)]
        for u, v, p in zip(rev_adj.row, rev_adj.col, rev_adj.data):
            adj_list[u].append((v, p))
        return adj_list



class SamplerForSIS:
    def __init__(self, graph_data):
        self.graph = graph_data
    def infer_generate(self, num_samples):
        rr_sets = []
        roots = np.random.randint(0, self.graph.num_nodes, num_samples)
        for root in roots:
            rr_set = {root}
            q = [root]
            while q:
                u = q.pop(0)
                if u >= len(self.graph.rev_adj_list): continue
                neighbors = self.graph.rev_adj_list[u]
                for v, prob in neighbors:
                    if v not in rr_set:
                        if random.random() <= prob:
                            rr_set.add(v)
                            q.append(v)
            rr_sets.append(rr_set)
        return rr_sets



class SISSimulator:
    def __init__(self, adj_matrix, beta=None, gamma=0.001):
        self.num_nodes = adj_matrix.shape[0]
        self.adj = adj_matrix
        self.gamma = gamma
        self.beta_matrix = adj_matrix

    def run_simulation(self, seeds, max_steps=100, num_simulations=100):
        seeds = list(seeds)
        if not seeds:
            return np.zeros(self.num_nodes, dtype=np.float32)
        total_infected_counts = np.zeros(self.num_nodes, dtype=np.float32)
        for _ in range(num_simulations):
            status = np.zeros(self.num_nodes, dtype=bool)
            status[seeds] = True
            
            for t in range(max_steps):
                if not np.any(status):
                    break
                
                recovery_prob = np.random.random(self.num_nodes)
                recovered = (status) & (recovery_prob < self.gamma)
                status[recovered] = False

                infected_indices = np.where(status)[0]
                if len(infected_indices) == 0:
                    continue
                
                row_idx, col_idx = self.adj.nonzero()
                probs = self.adj.data

                mask = status[row_idx]
                active_rows = row_idx[mask]
                active_cols = col_idx[mask]
                active_probs = probs[mask]
                
                random_rolls = np.random.random(len(active_probs))
                successful_transmissions = active_cols[random_rolls < active_probs]
                status[successful_transmissions] = True

            total_infected_counts += status.astype(np.float32)
            
        return total_infected_counts / num_simulations



def read_graph_topology(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if not lines: raise ValueError("Empty file")
        first_line = lines[0].split()
        if len(first_line) == 2:
            num_nodes = int(first_line[0])
            raw_edges = [list(map(int, line.split())) for line in lines[1:]]
        else:
            raw_edges = [list(map(int, line.split())) for line in lines]
            nodes = set()
            for u, v in raw_edges:
                nodes.add(u)
                nodes.add(v)
            num_nodes = max(nodes) + 1
    edges_np = np.array(raw_edges, dtype=np.int32)
    print(f"Graph Loaded: {num_nodes} nodes, {len(edges_np)} edges.")
    return num_nodes, edges_np



def build_full_adj_list(num_nodes, edges_np):
    adj = [[] for _ in range(num_nodes)]
    for u, v in edges_np:
        adj[u].append(v)
        adj[v].append(u) 
    return adj



def full_connected_component(num_nodes, edges_np):
    data = np.ones(len(edges_np))
    row = edges_np[:, 0]
    col = edges_np[:, 1]
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    n_components, labels = connected_components(csgraph=adj, directed=False, return_labels=True)
    
    if n_components == 1:
        return num_nodes, edges_np
    
    counts = np.bincount(labels)
    largest_label = np.argmax(counts)
    
    kept_nodes = np.where(labels == largest_label)[0]
    
    mask_u = np.isin(edges_np[:, 0], kept_nodes)
    mask_v = np.isin(edges_np[:, 1], kept_nodes)
    valid_edges = edges_np[mask_u & mask_v]
    
    new_u = np.searchsorted(kept_nodes, valid_edges[:, 0])
    new_v = np.searchsorted(kept_nodes, valid_edges[:, 1])
    new_edges = np.stack([new_u, new_v], axis=1)
    
    return len(kept_nodes), new_edges



def generate_bfs_subgraph_topology(num_nodes_total, edges_np, full_adj_list, keep_ratio):
    target_size = int(num_nodes_total * keep_ratio)
    
    sampled_nodes = set()
    while len(sampled_nodes) < target_size:
        start_node = random.randint(0, num_nodes_total - 1)
        while start_node in sampled_nodes:
            start_node = random.randint(0, num_nodes_total - 1)
            
        queue = deque([start_node])
        sampled_nodes.add(start_node)
        
        while queue and len(sampled_nodes) < target_size:
            u = queue.popleft()
            neighbors = full_adj_list[u]
            random.shuffle(neighbors)
            for v in neighbors:
                if v not in sampled_nodes:
                    sampled_nodes.add(v)
                    queue.append(v)
                    if len(sampled_nodes) >= target_size:
                        break
    kept_nodes = np.sort(list(sampled_nodes))
    
    mask_u = np.isin(edges_np[:, 0], kept_nodes)
    mask_v = np.isin(edges_np[:, 1], kept_nodes)
    sub_edges_raw = edges_np[mask_u & mask_v]
    
    new_u = np.searchsorted(kept_nodes, sub_edges_raw[:, 0])
    new_v = np.searchsorted(kept_nodes, sub_edges_raw[:, 1])
    sub_edges_remapped = np.stack([new_u, new_v], axis=1)

    final_nodes, final_edges = full_connected_component(len(kept_nodes), sub_edges_remapped)
    
    return final_nodes, final_edges



def build_prob_matrix_from_edges(num_nodes, edges_np):
    if len(edges_np) == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    counts = np.bincount(edges_np[:, 1])
    if len(counts) < num_nodes:
        counts = np.pad(counts, (0, num_nodes - len(counts)))
    in_degrees = counts.astype(np.float32)
    in_degrees[in_degrees == 0] = 1.0
    row = edges_np[:, 0]
    col = edges_np[:, 1]
    data = 1.0 / in_degrees[col]
    prob_matrix = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    return prob_matrix



def build_prob_matrix_from_edges_SIS(num_nodes, edges_np):
    if len(edges_np) == 0: return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    counts = np.bincount(edges_np[:, 1])
    if len(counts) < num_nodes: counts = np.pad(counts, (0, num_nodes - len(counts)))
    row = edges_np[:, 0]
    col = edges_np[:, 1]
    data = 0.001
    prob_matrix = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.float32)
    return prob_matrix



def process_single_subgraph_SIS(sub_adj, theta_feat=150, theta_label_sims=50):
    num_nodes = sub_adj.shape[0]
    
    num_seeds = random.randint(1, 40)
    seeds = set(np.random.choice(num_nodes, num_seeds, replace=False))
    
    sub_graph = GraphDataForSIS(sub_adj, num_nodes)
    feat_sampler = SamplerForSIS(sub_graph)
    rr_feat = feat_sampler.infer_generate(theta_feat)

    status_feat = np.ones(num_nodes, dtype=np.float32)
    for u in seeds: status_feat[u] = 0.0
    
    ris_counts = np.zeros(num_nodes, dtype=np.float32)
    valid_rr_feat = [rr for rr in rr_feat if rr.isdisjoint(seeds)]
    if len(valid_rr_feat) > 0:
        for rr in valid_rr_feat:
            for node in rr:
                ris_counts[node] += 1
        ris_counts = ris_counts / theta_feat
    
    features = np.stack([status_feat, ris_counts], axis=1)
    
    sis_sim = SISSimulator(sub_adj, gamma=0.001) 

    label = sis_sim.run_simulation(seeds, max_steps=50, num_simulations=theta_label_sims)
    
    return features, label, sub_adj



def process_single_subgraph(sub_adj, theta_feat, theta_label, infer_model='IC'):
    num_nodes = sub_adj.shape[0]
    sub_graph = GraphData(sub_adj, num_nodes)

    if infer_model == 'IC':
        sampler = Sampler_IC(sub_graph)
    elif infer_model == 'LT':
        sampler = Sampler_LT(sub_graph)
    
    num_seeds = random.randint(1, 40)
    seeds = set(np.random.choice(num_nodes, num_seeds, replace=False))
    
    rr_feat, _ = sampler.infer_generate(theta_feat)
    rr_label, rr_roots = sampler.infer_generate(theta_label)
    
    covered_nodes = set(seeds)
    covered_rr_indices_label = []
    for i, rr in enumerate(rr_label):
        if not rr.isdisjoint(seeds):
            covered_nodes.add(rr_roots[i])
            covered_rr_indices_label.append(i)
            
    status_feat = np.ones(num_nodes, dtype=np.float32)
    for u in covered_nodes:
        status_feat[u] = 0.0
        
    valid_rr_feat = [rr for rr in rr_feat if rr.isdisjoint(seeds)]
    covered_indices_set = set(covered_rr_indices_label)
    valid_rr_label = [rr for i, rr in enumerate(rr_label) if i not in covered_indices_set]


    def count_frequency(rr_list, total_rr_count):
        counts = np.zeros(num_nodes, dtype=np.float32)
        if total_rr_count == 0: return counts
        for rr in rr_list:
            for node in rr:
                counts[node] += 1
        return counts / total_rr_count

    ris_feat = count_frequency(valid_rr_feat, theta_feat)
    label = count_frequency(valid_rr_label, theta_label)
    
    features = np.stack([status_feat, ris_feat], axis=1)
    
    return features, label, sub_adj


def generate_dataset(file_name, num_subgraphs=500, infer='IC'):
    file_txt_path = f'data/{file_name}.txt'
    num_nodes_total, edges_total_np = read_graph_topology(file_txt_path)
    full_adj_list = build_full_adj_list(num_nodes_total, edges_total_np)
    dataset = []

    generated_count = 0
    pbar = tqdm(total=num_subgraphs)
    
    while generated_count < num_subgraphs:
        keep_ratio = random.uniform(0.7, 1.0)
        
        num_sub_nodes, sub_edges = generate_bfs_subgraph_topology(
            num_nodes_total, edges_total_np, full_adj_list, keep_ratio
        )
        
        if num_sub_nodes < 500:
            continue
        if infer=='SIS':
            sub_adj = build_prob_matrix_from_edges_SIS(num_sub_nodes, sub_edges)
            features, label, adj = process_single_subgraph_SIS(sub_adj, theta_feat=150, theta_label=50 )
        else:
            sub_adj = build_prob_matrix_from_edges(num_sub_nodes, sub_edges)
            features, label, adj = process_single_subgraph(sub_adj, theta_feat=200, theta_label=10000, infer_model=infer )
    
        if np.max(label) < 0.002:
            continue

        dataset.append({'adj': adj, 'features': features, 'label': label})
        generated_count += 1
        pbar.update(1)
        
    pbar.close()
    
    os.makedirs('inidata', exist_ok=True)
    save_path = f'inidata/{file_name}_data_{infer}.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Saved to {save_path}")


