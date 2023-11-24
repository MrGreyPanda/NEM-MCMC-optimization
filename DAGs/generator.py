import os
import dot
import rnd_dag_gen
import graphviz
import os

def create_graph_from_dot(input_file, output_file):
    graph = graphviz.Source.from_file(input_file)
    # Render the graph and save it as a PDF
    if os.path.exists(output_file):
        os.remove(output_file)
    graph.render(outfile=output_file, view=False, format="pdf")

def dot_gen(i, input_file, output_file):
    if not os.path.exists(output_file):
        dot.generate_dot_language(input_file, output_file)
            
def graph_gen(i, input_file, output_file):
    create_graph_from_dot(input_file, output_file)
    print("Done!")
        
if __name__ == "__main__":
    curr_dir = os.getcwd()
    for i in range(20):
        print(f"Creating data {i}")
        if not os.path.exists(curr_dir + f"/networks/network{i}"):
            os.makedirs(curr_dir + f"/networks/network{i}", exist_ok=True)
            rnd_dag_gen.gen_rnd_dag(i, output_file_path=f"networks/network{i}/", min_ranks = 4, max_ranks=8)
            dot_gen(i, input_file=f"networks/network{i}/network{i}.csv", output_file=f"networks/network{i}/network{i}.dot")
            graph_gen(i, input_file=f"networks/network{i}/network{i}.dot", output_file=f"networks/network{i}/network{i}.pdf")
            # transitively reduced networks
            dot_gen(i, input_file=f"networks/network{i}/network{i}_red.csv", output_file=f"networks/network{i}/network{i}_red.dot")
            graph_gen(i, input_file=f"networks/network{i}/network{i}_red.dot", output_file=f"networks/network{i}/network{i}_red.pdf")
        
