import torch

import json
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

from graph_generators import graph_gen
from apsp import FW_GPU, R_Kleene

### Converter to nx Graph
def convert_to_nxg(H):

    H_sparse = H.to_sparse()
    edge_index, weight = H_sparse.indices().cpu().numpy(), H_sparse.values().cpu().numpy()
    edge_and_weights = [(edge[0], edge[1], w) for edge, w in zip(edge_index.T, weight.T)]

    g = nx.Graph()
    g.add_weighted_edges_from(edge_and_weights)

    return g

#### BENCHMARKING #######
def benchmark(num_exps=500, max_nodes=1000, max_density=100, do_nx=False):

    # cuda0 = torch.device('cuda:0')
    # cuda1 = torch.device('cuda:1')
    cuda = [ torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

    num_nodess = torch.randint(4,max_nodes,size=(1,num_exps)).flatten()
    densities = torch.randint(0,max_density,size=(1,num_exps)).flatten() * torch.rand(num_exps)
    num_edgess = num_exps * [None]

    fw_cc, rkl_cc, nx_cc = num_exps * [None], num_exps * [None], num_exps * [None]
    inconsistencies_kl_fw, inconsistencies_nx_rkl, inconsistencies_nx_fw = num_exps * [None], num_exps * [None], num_exps * [None]

    for exp, (num_nodes, density) in enumerate(zip(num_nodess, densities)):
        print('iter:', exp)
        H, num_edges = graph_gen(num_nodes=num_nodes, density_controller=density) 
        num_edgess[exp] = num_edges

        H_clone = H.to(cuda[-1]) #H.clone().to(cuda1)

        curr = time.time()
        H_fw = FW_GPU(H_clone)['cost']
        fw_cc[exp] = time.time() - curr
        H_fw = H_fw.to(cuda[0])
        #H_fw = torch.round(H_fw) #, decimal=2)

        curr = time.time()
        H_rkl = R_Kleene(H_clone)
        rkl_cc[exp] = time.time() - curr
        H_rkl = H_rkl.to(cuda[0])
        #H_rkl = torch.round(H_rkl) #, decimal=2)

        if do_nx:
            g = convert_to_nxg(H)
            curr = time.time()
            H_nx = nx.floyd_warshall_numpy(g)
            nx_cc[exp] = time.time() - curr

            H_nx = torch.Tensor(H_nx).to(H.device)

            inconsistency_nx_fw = torch.nn.functional.mse_loss(H_nx, H_fw)
            inconsistencies_nx_fw[exp] = inconsistency_nx_fw.item()
            
            inconsistency_nx_rkl = torch.nn.functional.mse_loss(H_nx, H_rkl)
            inconsistencies_nx_rkl[exp] = inconsistency_nx_rkl.item()

        #assert (H_rkl == H_fw).all(), 'Inconsistent Results between GPU algs!'
        inconsistency_kl_fw = torch.nn.functional.mse_loss(H_rkl, H_fw)
        inconsistencies_kl_fw[exp] = inconsistency_kl_fw.item()


    # Format
    num_nodess = num_nodess.cpu().numpy()
    num_edgess = np.array(num_edgess)

    rkl_cc = np.array(rkl_cc)
    fw_cc = np.array(fw_cc)
    nx_cc = np.array(nx_cc)

    inconsistencies_kl_fw = np.array(inconsistencies_kl_fw)
    inconsistencies_nx_fw = np.array(inconsistencies_nx_fw)
    inconsistencies_nx_rkl = np.array(inconsistencies_nx_rkl)

    return {'graph_specs': [num_nodess, num_edgess], 'time_costs': [rkl_cc, fw_cc, nx_cc], 'inconsistencies': [inconsistencies_kl_fw, inconsistencies_nx_fw, inconsistencies_nx_rkl]}

def sort_by_and_plot(big_dict, sort_by='nodes', do_nx=False, exp_slice=None): # num_nodess, num_edgess, rkl_cc, fw_cc, inconsistencies,
    num_nodess, num_edgess = big_dict['graph_specs']
    rkl_cc, fw_cc, nx_cc = big_dict['time_costs']
    inconsistencies_kl_fw, inconsistencies_nx_fw, inconsistencies_nx_rkl = big_dict['inconsistencies']

    if exp_slice is None:
        start, stop = 0, num_nodess.shape[0]
    else:
        start, stop = exp_slice

    # Sort
    if sort_by == 'nodes':
        sorting_idxs = np.argsort(num_nodess)
    elif sort_by == 'edges':
        sorting_idxs = np.argsort(num_edgess)
    num_nodess = num_nodess[sorting_idxs][start:stop]
    num_edgess = num_edgess[sorting_idxs][start:stop]
    rkl_cc = rkl_cc[sorting_idxs][start:stop]
    fw_cc = fw_cc[sorting_idxs][start:stop]
    nx_cc = nx_cc[sorting_idxs][start:stop]
    inconsistencies_kl_fw = inconsistencies_kl_fw[sorting_idxs][start:stop]
    inconsistencies_nx_fw = inconsistencies_nx_fw[sorting_idxs][start:stop]
    inconsistencies_nx_rkl = inconsistencies_nx_rkl[sorting_idxs][start:stop]

    # Plots
    plt.figure()
    plt.plot(num_nodess, label='num nodes of graph')
    plt.plot(np.sqrt(num_edgess), label='num edges of graph in sqrt scale')
    plt.xlabel('graph/exp number')
    plt.ylabel('amount')
    plt.title('Showing graph statistics of each graph-experimental-sample')
    plt.legend()
    plt.show()

    plt.figure()
    graph_densities = num_edgess / ((num_nodess**2) - num_nodess)
    plt.plot(graph_densities, label='graph density')
    plt.xlabel('graph/exp number')
    plt.ylabel('Graph Density')
    plt.title('Showing connectivity/density of each graph-experiment-sample')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(fw_cc, label='FW_GPU() time costs')
    plt.plot(rkl_cc, label='R_Kleene_GPU() time costs')
    if do_nx:
        plt.plot(nx_cc, label='FW_CPU_NX() time costs')
    plt.xlabel('graph/exp number')
    plt.ylabel('time cost in seconds')
    plt.title('Comparing Time Costs of APSP Implementations')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(inconsistencies_kl_fw, label='between FW_GPU() and R_Kleene_GPU()')
    if do_nx:
        plt.plot(inconsistencies_nx_fw, label='between FW_CPU_NX() and FW_GPU()')
        plt.plot(inconsistencies_nx_rkl, label='between FW_CPU_NX() and R_Kleene_GPU()')
    plt.xlabel('graph/exp number')
    plt.ylabel('Average Mean Squared Error(MSE)')
    plt.title('Quantifying Inconsitencies Between APSP Implementations')
    plt.legend()
    plt.show()


def save_to_json(big_dict, fname='big_dict.json'):
    new_big_dict = {}
    new_big_dict['graph_specs'] = [big_dict['graph_specs'][0].tolist(), big_dict['graph_specs'][1].tolist()]
    new_big_dict['time_costs'] = [big_dict['time_costs'][0].tolist(), big_dict['time_costs'][1].tolist(), big_dict['time_costs'][2].tolist()]
    new_big_dict['inconsistencies'] =  [big_dict['inconsistencies'][0].tolist(), big_dict['inconsistencies'][1].tolist(), big_dict['inconsistencies'][2].tolist()]

    # write result big_dict to json
    with open(fname,"w") as f:
        json.dump(new_big_dict,f)

def load_from_json(fname='big_dict.json'):
    with open(fname) as json_file:
        big_dict = json.load(json_file)

    new_big_dict = {}
    new_big_dict['graph_specs'] = [np.array(big_dict['graph_specs'][0]), np.array(big_dict['graph_specs'][1])]
    new_big_dict['time_costs'] = [np.array(big_dict['time_costs'][0]), np.array(big_dict['time_costs'][1]), np.array(big_dict['time_costs'][2])]
    new_big_dict['inconsistencies'] =  [np.array(big_dict['inconsistencies'][0]), np.array(big_dict['inconsistencies'][1]), np.array(big_dict['inconsistencies'][2])]

    return new_big_dict