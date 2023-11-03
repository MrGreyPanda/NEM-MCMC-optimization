import csv
import os

def generate_dot_language(csv_file_path, output_file_path):
    # Read in the CSV file
    with open(csv_file_path, 'r') as f:
        print(f"Reading {csv_file_path}")
        reader = csv.reader(f)
        data = []
        for row in reader:
            data.append(row)
        data.pop(0)
        data.pop()
        data.pop()
        # Create the DOT language string
        dot_string = 'digraph {\n'
        # Add the edges to the DOT language string
        for edge in data:
            dot_string += '    {} -> {};\n'.format(edge[0], edge[1])
        # Close the DOT language string
        dot_string += '}'
        
        # Write the DOT language string to the output file
    with open(output_file_path, 'w') as file:
        file.write(dot_string)
    print(f"Generated {output_file_path}")
            
def generate_dot_from_matrix(adj_matrix, output_file):
    dot_string = 'digraph {\n'
    # Add the edges to the DOT language string
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] != 0:
                dot_string += '    {} -> {};\n'.format(i, j)
    # Close the DOT language string
    dot_string += '}'
    
    # Write the DOT language string to the output file
    with open(output_file, 'w') as file:
        file.write(dot_string)
    print(f"Generated {output_file}")