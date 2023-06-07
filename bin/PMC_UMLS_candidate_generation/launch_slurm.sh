#!/bin/bash
# Will launch jobs for every number between the two specified arguments (inclusive).
# If only one argument is provided, it will be interpreted as the last number to
# process and will start at 1

for i in $(seq $1 $2)
do
	sbatch generate_candidates_$i.slurm
done