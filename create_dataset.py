import matplotlib.pyplot as plt
from preprocessors import BDIPreProcessor, DecompressionPreProcessor

data_folder = "/Volumes/Elements/Camera Obscura Backup Footage/"

bdi_proc = BDIPreProcessor(
    data_folder,
    "./data/processed_bdi",
    output_size = (256, 448),
    stride = 7,
    reference_ind = 3,
    size_targ = 1e5,
    ns_retrieval = 50,
    search_suffix='_001.mov'
)
bdi_proc.process()

decomp_proc = DecompressionPreProcessor(
    data_folder,
    "./data/processed_decomp",
    output_size=(256, 448),
    stride=7,
    reference_ind=3,
    size_targ=1e5,
    ns_retrieval=50,
    search_suffix='_001.mov'
)
decomp_proc.process()



