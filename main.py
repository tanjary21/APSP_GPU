import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import json

from utilss import benchmark, save_to_json, load_from_json, sort_by_and_plot
from apsp import FW_GPU, R_Kleene

if __name__ == "__main__":
    
    print("\nStarting...\n")
    do_nx = True
    big_dict = benchmark(num_exps=10, max_nodes=1000, max_density=100, do_nx=do_nx)
    save_to_json(big_dict)