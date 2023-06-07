#!/usr/bin/bash
if [ $# -gt 0 ] && [ "$1" -lt "90" ]
then
    STEP=$1
else
    STEP=90
fi

#for I in {0..511..$STEP}
for I in $(seq 0 $STEP 511)
do
    ./convert_scispacy_jsonl_to_cand_slurmcreator.sh $I $(( $I + $STEP - 1 ))
done
./convert_scispacy_jsonl_to_cand_slurmcreator.sh $I 511
