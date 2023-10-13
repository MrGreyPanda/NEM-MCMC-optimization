import csv
import os

def generate_dot_language():
    # If output file path is not provided, use the same directory and filename as the input file
    i = 1
    csv_file_path = f"csv/network{i}.csv"
    while os.path.exists(csv_file_path):
        csv_file_path = f"csv/network{i}.csv"
        dot_string = ''
        output_file_path = f"dot/network{i}.dot"
        i += 1
        if not os.path.exists(output_file_path):
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
        if i >= 100:
            break
            
generate_dot_language()