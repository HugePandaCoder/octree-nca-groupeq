
prostate_dataset_config = {
    'experiment.dataset.img_path': r"jkalkhof/Data/Prostate_MEDSeg/imagesTr/",
    'experiment.dataset.label_path': r"jkalkhof/Data/Prostate_MEDSeg/labelsTr/",
    'experiment.dataset.keep_original_scale': True,
    'experiment.dataset.rescale': True,
    'experiment.dataset.input_size': [320, 320, 24],
    'experiment.dataset.patchify': False,

    'experiment.dataset.split_file': r"clmn1/octree_study/Experiments/Prostate49_OctreeNCA3D/data_split.pkl", 

    'model.input_channels': 1,

    'trainer.num_steps_per_epoch': None,
    'trainer.batch_size': 3,
    'trainer.batch_duplication': 1,
    }