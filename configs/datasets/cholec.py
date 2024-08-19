cholec_dataset_config = {
    'experiment.dataset.img_path': r"clmn1/data/cholecseg8k_preprocessed_2/",
    'experiment.dataset.label_path': r"clmn1/data/cholecseg8k_preprocessed_2/",
    'experiment.dataset.keep_original_scale': True,
    'experiment.dataset.rescale': True,
    'experiment.dataset.input_size': [240, 432, 80],

    'experiment.dataset.split_file': r"clmn1/octree_study/Experiments/cholec_seg_5_OctreeNCA3D/data_split.pkl", 

    'model.input_channels': 3,

    'trainer.num_steps_per_epoch': None,
    'trainer.batch_size': 2,
    'trainer.batch_duplication': 1,
}