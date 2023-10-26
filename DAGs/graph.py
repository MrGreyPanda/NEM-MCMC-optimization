import graphviz
import os

def create_graph_from_dot():
    i = 1
    print("Creating graphs...")
    input_file = f"dot/network{i}.dot"
    while os.path.exists(input_file):
        output_file = f"graph/network{i}.pdf"
        # if os.path.exists(output_file):
        #     i +=1
        #     continue
        # Create a Graph object from the .dot file
        graph = graphviz.Source.from_file(input_file)
        # Render the graph and save it as a PDF
        graph.render(outfile=output_file, view=False)

        
        i += 1
        input_file = f"dot/network{i}.dot"
        if i > 100:
            print("Done!")
            break
        
create_graph_from_dot()