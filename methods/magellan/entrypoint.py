import os
import argparse
import pathtype

import py_entitymatching as em

from transform import transform_input, transform_output


parser = argparse.ArgumentParser(description='Benchmark a dataset with a method')
parser.add_argument('input', type=pathtype.Path(readable=True), nargs='?', default='/data',
                    help='Input directory containing the dataset')
parser.add_argument('output', type=pathtype.Path(writable=True), nargs='?', default='/data/output',
                    help='Output directory to store the output')
parser.add_argument('-t', '--temp', type=pathtype.Path(writable_or_creatable=True), nargs='?', default='/tmpdir',
                    help='A folder to store temporary files')
parser.add_argument('-r', '--recall', type=int, nargs='?',
                    help='Recall value for the algorithm')
parser.add_argument('-e', '--epochs', type=int, nargs='?',
                    help='Number of epochs for the algorithm')

args = parser.parse_args()

print("Hi, I'm Magellan entrypoint!")
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

# Create a temporary directory to store intermediate files
# FIXME: Could be removed if not needed (e.g. if the method does not require any intermediate files or data stored in memory)
temp_input = os.path.join(args.temp, 'input')
temp_output = os.path.join(args.temp, 'output')

# Step 1. Convert input data into the format expected by the method
# FIXME: feel free to change the number of arguments, or store the data in variables and return them from the function
transform_input(args.input, temp_input)
print("Method input: ", os.listdir(temp_input))

# Step 2. Run the method
# FIXME: implement the matching algorithm
tableA = em.read_csv_metadata(os.path.join(args.input, 'tableA.csv'), key='id')
tableB = em.read_csv_metadata(os.path.join(args.input, 'tableB.csv'), key='id')

ab = em.AttrEquivalenceBlocker()
C1 = ab.block_tables(tableA, tableB, 'paper year', 'paper year', l_output_attrs=['title', 'authors', 'paper year'], r_output_attrs=['title', 'authors', 'paper year'])
print("Method output: ", os.listdir(temp_output))

# Step 3. Convert the output into a common format
# FIXME: feel free to change the number of arguments, but make sure to persist the data to the output directory
transform_output(temp_output, args.output)
print("Final output: ", os.listdir(args.output))
