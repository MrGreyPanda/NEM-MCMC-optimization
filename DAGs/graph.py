import graphviz
import os

def create_graph_from_dot(input_file, output_file):
    graph = graphviz.Source.from_file(input_file)
    # Render the graph and save it as a PDF
    if os.path.exists(output_file):
        os.remove(output_file)
    graph.render(outfile=output_file, view=False, format="pdf")
