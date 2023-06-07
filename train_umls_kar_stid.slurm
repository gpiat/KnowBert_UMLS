#!/bin/bash
#SBATCH -e log_kar_train_STID_20220610.err
#SBATCH -o log_kar_train_STID_20220610.out
# Choisir nb noeuds
#SBATCH -N 1
## nombre processus
#SBATCH -n 1
## nombre de threads
##SBATCH -c
## le noeud entier rien que pour moi
##SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
# Choisir partition (commande `sinfo` pour voir les differentes partitions)
#SBATCH -p gpup100,gpup6000,gpuv100
##,gpup5000short
# Nb GPU que je veux
#SBATCH --gres=gpu:1
# Mail pour etre informe de l'etat de votre job
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=gael.bernard@cea.fr
# Nom de votre job afficher dans la lise par squeue
#SBATCH --job-name=umls_STID_kar_train
# Limite du nb de CPU

# Chargment de vos modules
#
echo "Begin loading module..."
module load anaconda/3.2021.05 cuda/10.0


# Affiche la machine(s)
echo "Begin on machine :"
hostname
# Pour voir si le job s'exécute bien sur les cartes graphiques
# Affiche l'utilisation des cartes graphiques au début de l'exécution
# nvidia-smi


OUTPUT_DIRECTORY=`pwd`'/umls_data/2022_05_25_kar_train_stid/'
rm -r $OUTPUT_DIRECTORY*

# this config uses scispacy as entity linker and uses STIDs as entity vocabulary (as opposed to preferred concept names) and entity IDs
srun --gres=gpu:1 allennlp train -s $OUTPUT_DIRECTORY --file-friendly-logging --include-package kb.include_all training_config/pretraining/knowbert_umls_stid_linker.jsonnet

wait

echo "Done."
# ./ Fin
