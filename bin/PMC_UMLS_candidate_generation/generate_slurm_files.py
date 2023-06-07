""" Will generate slurm files for all file numbers between the first and
    second command line arguments (inclusive).
    Each slurm file launches candidate generation for one dataset shard.
    Each job reserves one node exclusively (because of the amount of
    memory required) with a NICE value proportional to its number to
    ensure lower-numbered dataset shards have priority over higher ones.
"""
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('start', metavar='start', type=int)
parser.add_argument('end', metavar='end', type=int)

args = parser.parse_args()

for i in range(args.start, args.end + 1):
    contents = f'''#!/bin/bash
#SBATCH -e logs/gen_candidates_{i}.err
#SBATCH -o logs/gen_candidates_{i}.out

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH -p allcpu
#SBATCH --job-name={i}_GenCand
#SBATCH --nice={int(i/4)}
#SBATCH --exclusive

echo "Begin on machine :"
hostname

OUTPUT_DIRECTORY="/scratch/gpiat/pmc/oa_bulk_bert_512/cand_files/"
[ ! -d $OUTPUT_DIRECTORY ] && mkdir -p $OUTPUT_DIRECTORY
rsync -a -u gpiat@factoryia02:/media/pmc/oa_bulk_bert_512/ /scratch/gpiat/pmc/oa_bulk_bert_512/

srun -n 1 python generate_UMLS_candidates.py --out_dir $OUTPUT_DIRECTORY {i} &
wait

rsync -a -u /scratch/gpiat/pmc/oa_bulk_bert_512/ gpiat@factoryia02:/media/pmc/oa_bulk_bert_512/

echo "Done."

'''
    with open(f'generate_candidates_{i}.slurm', 'w') as f:
        print(contents, file=f)
