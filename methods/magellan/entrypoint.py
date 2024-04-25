import os

import py_entitymatching as em

from transform import transform_dataset
from parser import entry_args


args = entry_args().parse_args()

print("Hi, I'm Magellan entrypoint!")
print("Input directory: ", os.listdir(args.input))
print("Output directory: ", os.listdir(args.output))

transform_dataset(args.input, args.output)
print("Output directory (after transform): ", os.listdir(args.output))

tableA = em.read_csv_metadata(os.path.join(args.input, 'tableA.csv'), key='id')
tableB = em.read_csv_metadata(os.path.join(args.input, 'tableB.csv'), key='id')

ab = em.AttrEquivalenceBlocker()
C1 = ab.block_tables(tableA, tableB, 'paper year', 'paper year', l_output_attrs=['title', 'authors', 'paper year'], r_output_attrs=['title', 'authors', 'paper year'])
