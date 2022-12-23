import torch

#### Random Graph Generator ####
def graph_gen(num_nodes=10, density_controller=.9, cost_magnitude_controller=100):
    cuda0 = torch.device('cuda:0')
    dc= density_controller # less than one makes it sparser, greater than one makes it denser
    cmc = cost_magnitude_controller # controls the magnitude of the costs

    dense_adj = torch.bernoulli(torch.clamp(dc*torch.rand(num_nodes, num_nodes), max=1.0)).to(cuda0).int()
    dense_adj[torch.arange(num_nodes), torch.arange(num_nodes)] = 0
    num_edges = dense_adj.sum() # count the total number of edges generated in this graph

    costs = (cmc * torch.rand(dense_adj.shape).to(cuda0))#.int()
    costs[~dense_adj.bool()] = torch.inf 
    costs[torch.arange(num_nodes), torch.arange(num_nodes)] = 0
    
    return costs, num_edges.item()