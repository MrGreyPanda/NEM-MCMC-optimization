import random
import os
import csv

def gen_rnd_dag(min_per_rank=1, max_per_rank=5, min_ranks=2, max_ranks=5, percent=35, min_end_nodes=10, max_end_nodes=20):
    nodes = 0
    ranks = min_ranks + random.randint(0, max_ranks - min_ranks + 1)
    i = 1
    output_file = f"csv/network{i}.csv"
    while os.path.exists(output_file):
        i += 1
        output_file = f"csv/network{i}.csv"
    data = []
    for i in range(ranks):
        # New nodes of 'higher' rank than all nodes generated till now.
        new_nodes = min_per_rank + random.randint(0, max_per_rank - min_per_rank + 1)

        # Edges from old nodes ('nodes') to new ones ('new_nodes').
        for j in range(nodes):
            for k in range(new_nodes):
                if random.randint(0, 99) < percent:
                    data.append([j, k + nodes]) # An Edge.

        nodes += new_nodes # Accumulate into old node set.
    num_end_nodes = random.randint(min_end_nodes, max_end_nodes)
    end_nodes = []
    probs = [0.05, 0.08]
    first_line = [nodes, num_end_nodes]
    for i in range(num_end_nodes):
        end_nodes.append(random.randint(0, nodes-1))
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerows(data)
        writer.writerow(end_nodes)
        writer.writerow(probs)
        
gen_rnd_dag()
