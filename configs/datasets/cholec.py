def get_cholec_dataset_config():
    raise NotImplementedError
    return {
    'trainer.num_steps_per_epoch': None,
    'trainer.batch_size': 2,
    'trainer.batch_duplication': 1,
    
    #'experiment.dataset.seed': 42,
    }
