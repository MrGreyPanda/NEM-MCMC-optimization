import random
import os
import csv


def transitive_closure(a):
    """
    Computes the transitive closure of a given set of relations.
    Implementation from: https://stackoverflow.com/questions/8673482/transitive-closure-python-tuples

    Args:
        a (set): A set of relations.

    Returns:
        set: The transitive closure of the input set of relations.
    """
    closure = set(a)
    while True:
        new_relations = set((x,w) for x,y in closure for q,w in closure if q == y)
        closure_until_now = closure | new_relations
        if closure_until_now == closure:
            break 
        closure = closure_until_now

    return closure

def transitive_reduction(a):
    """
    Computes the transitive reduction of a given set of relations.
    Implementation from: https://stackoverflow.com/questions/8673482/transitive-closure-python-tuples

    Args:
        a (set): A set of relations.

    Returns:
        set: The transitive reduction of the input set of relations.
    """
    reduction = set(a)
    while True:
        new_relations = set((x,w) for x,y in reduction for q,w in reduction if q == y and (x,w)  in reduction)
        reduction_until_now = reduction - new_relations
        if reduction_until_now == reduction:
            break 
        reduction = reduction_until_now
    return reduction

def gen_rnd_dag(i, output_file_path, min_per_rank=2, max_per_rank=4, min_ranks=3, max_ranks=5, percent=40, probs=[0.05, 0.1]):
    """
    Generates a random directed acyclic graph (DAG) and writes it to a CSV file.
    Implementation follows Algorithm from: https://stackoverflow.com/questions/12790337/generating-a-random-dag

    Args:
        min_per_rank (int): Minimum number of nodes per rank.
        max_per_rank (int): Maximum number of nodes per rank.
        min_ranks (int): Minimum number of ranks.
        max_ranks (int): Maximum number of ranks.
        percent (int): Probability of an edge between two nodes.
        min_end_nodes (int): Minimum number of end nodes.
        max_end_nodes (int): Maximum number of end nodes.

    Returns:
        None
    """
    nodes = 0
    ranks = min_ranks + random.randint(0, max_ranks - min_ranks + 1)
    output_file = output_file_path + f"network{i}.csv"
    output_file_red = output_file_path + f"network{i}_red.csv"
    data = []
    for i in range(ranks):
        # New nodes of 'higher' rank than all nodes generated till now.
        new_nodes = min_per_rank + random.randint(0, max_per_rank - min_per_rank + 1)

        # Edges from old nodes ('nodes') to new ones ('new_nodes').
        for j in range(nodes):
            for k in range(new_nodes):
                if random.randint(0, 100) < percent:
                    data.append((j, k + nodes)) # An Edge.

        nodes += new_nodes # Accumulate into old node set.
    data_reduced = transitive_reduction(data)
    for i in range(len(data)):
        data = transitive_closure(data)
    min_end_nodes = 8 * nodes
    max_end_nodes = 12 * nodes
    num_end_nodes = random.randint(min_end_nodes, max_end_nodes)
    end_nodes = []
    first_line = [nodes, num_end_nodes]
    i = 0
    for _ in range(num_end_nodes):
        end_nodes.append(random.randint(0, nodes-1))
                
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerows(data)
        writer.writerow(end_nodes)
        writer.writerow(probs)
    with open(output_file_red, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerows(data_reduced)
        writer.writerow(end_nodes)
        writer.writerow(probs)
        
