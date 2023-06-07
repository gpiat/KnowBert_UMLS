#!/usr/bin/bash

DIR="$( dirname $0 )"
START=`printf %03d $1`
END=`printf %03d $2`
OUTFILE=$DIR/convert_scispacy_jsonl_to_cand_"$START"_"$END".slurm

sed s/START/$START/g $DIR/convert_scispacy_jsonl_to_cand_template.slurmtemp > $OUTFILE
sed -i s/END/$END/g $OUTFILE

sbatch $OUTFILE
