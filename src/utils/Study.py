from .Experiment import Experiment


class Study():
    r"""This class handles:
            - Running multiple experiments at once
    """

    def __init__(self) -> None:
        self.experiments = []

    def add_experiment(self, experiment: Experiment) -> None:
        r"""Add an experiment to the study
            #Args
                experiment (Experiment): Experiment to add
        """
        self.experiments.append(experiment)

    def run_experiments(self) -> None:
        r"""Run all experiments
        """
        for experiment in self.experiments:
            experiment.agent.train(experiment.data_loader, experiment.loss_function)