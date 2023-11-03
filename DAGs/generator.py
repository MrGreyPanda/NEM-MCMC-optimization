import os
import dot
import graph
import rnd_dag_gen

def dot_gen():
    # If output file path is not provided, use the same directory and filename as the input file
    i = 1
    csv_file_path = f"csv/network{i}.csv"
    while os.path.exists(csv_file_path):
        if i >= 100:
            break
        csv_file_path = f"csv/network{i}.csv"
        dot_string = ''
        output_file_path = f"dot/network{i}.dot"
        i += 1
        if not os.path.exists(output_file_path):
            dot.generate_dot_language(csv_file_path, output_file_path)
            
def graph_gen():
    i = 1
    print("Creating graphs...")
    input_file = f"dot/network{i}.dot"
    while os.path.exists(input_file):
        output_file = f"graph/network{i}.pdf"
        graph.create_graph_from_dot(input_file, output_file)
        i += 1
        input_file = f"dot/network{i}.dot"
        if i > 100:
            print("Done!")
            break
        
if __name__ == "__main__":
    # rnd_dag_gen.gen_rnd_dag()
    # dot_gen()
    graph_gen()
        
