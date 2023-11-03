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

def gen_rnd_dag(min_per_rank=1, max_per_rank=5, min_ranks=2, max_ranks=5, percent=35, max_end_nodes=30):
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
                    data.append((j, k + nodes)) # An Edge.

        nodes += new_nodes # Accumulate into old node set.
    data = transitive_closure(data)
    min_end_nodes = int(1.5 * nodes)
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
