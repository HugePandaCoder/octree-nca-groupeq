def get_peso_dataset_config():
    return {
        'experiment.dataset.img_path': r"/local/scratch/PESO/peso_training",
        'experiment.dataset.label_path': r"/local/scratch/PESO/peso_training",
        'experiment.dataset.keep_original_scale': True,
        'experiment.dataset.rescale': True,
        'experiment.dataset.input_size': [320, 320],
        'experiment.dataset.img_level': 1,

        'trainer.num_steps_per_epoch': 200,
        'trainer.batch_size': 3,
        'trainer.batch_duplication': 1,
    }