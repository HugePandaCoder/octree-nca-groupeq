segmentation_task_config = {
    'experiment.task': "segmentation",
    'experiment.task.score': ["src.scores.DiceScore.DiceScore", "src.scores.IoUScore.IoUScore"],
    'trainer.losses': ["src.losses.DiceBCELoss.DiceBCELoss"],
    'trainer.loss_weights': [1.],
    }