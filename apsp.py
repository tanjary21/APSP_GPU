import torch

#### FLOYD WARSHALL ALGS ####
def FW_GPU(H):
    found = False
    while not found:
        H_prime, pred = FW_iter(H)
        if (H_prime == H).all():
            found=True
        H = H_prime
    return {'cost': H, 'pred': pred}

def FW_iter(costs):
    threeD = costs + costs.T.unsqueeze(1)
    updated_costs, updated_vias = torch.min(threeD, dim=2)
    updated_costs, updated_vias = updated_costs.T, updated_vias.T
    return updated_costs, updated_vias

#### R-KLEENE ALGS ####
def min_plus_GPU(A, B, C=None):
    threeD = A + B.T.unsqueeze(1)
    updated_costs, updated_vias = torch.min(threeD, dim=2)
    updated_costs, updated_vias = updated_costs.T, updated_vias.T
    if C is None:
        return updated_costs #, updated_vias
    else:
        return torch.minimum(updated_costs, C) #, updated_vias

def R_Kleene(H):
    h,w = H.shape
    assert h == w, 'The input cost matrix should be square'
    assert (H >= 0).any(), "The input cost matrix shouldn't have negative costs "
    assert (torch.diagonal(H) == 0).all(), 'The input cost matrix should have a zero diagonal'

    # base case
    if h <= 2:
        return H

    # recursive case
    cut_point = h//2
    A = H[:cut_point, :cut_point]
    B = H[:cut_point, cut_point:]
    C = H[cut_point:, :cut_point]
    D = H[cut_point:, cut_point:]

    A = R_Kleene(A);          # recursive call, compute path lengths within A
    B = min_plus_GPU(A,B);    # B = A*B;       now B includes paths through A
    C = min_plus_GPU(C,A);    # C = C*A;       now C includes paths through A
    D = min_plus_GPU(C,B,D);  # D = D + C*B;   now D includes paths through A
    D = R_Kleene(D);          # recursive call, compute path lengths within D
    B = min_plus_GPU(B,D);    # B = B*D;       now B includes paths through D
    C = min_plus_GPU(D,C);    # C = D*C:       now C includes paths through D
    A = min_plus_GPU(B,C,A);  # A = A + B*C;   now A includes paths through D

    result = torch.cat([ torch.cat([A, B], dim=1), torch.cat([C, D], dim=1) ], dim=0)
    return result