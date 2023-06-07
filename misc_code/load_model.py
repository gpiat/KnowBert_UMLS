from allennlp.models.archival import load_archive
from kb import include_all
from allennlp.common.util import prepare_environment

#tar -czvf umls_data/knowbert_coarse_tuned_pmc.tar.gz -C `pwd`/umls_data/knowbert_coarse_tuned_pmc/ .
cuda_device = -1
overrides = ""
weights_file = ""  # "bkp_logs/knowbert_umls_finetune_i2b2/best.th"
# archive_file = 'bkp_logs/2021_06_10__01/kar_train/model.tar.gz'
archive_file = "umls_data/knowbert_coarse_tuned_pmc.tar.gz"
archive = load_archive(archive_file, cuda_device, overrides, weights_file)
config = archive.config
prepare_environment(config)
model = archive.model
model.eval()
