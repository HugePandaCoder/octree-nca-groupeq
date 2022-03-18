from src.utils.helper import saveNiiGz

import nibabel as nib
import numpy as np
import os
import torch

class BaseAgent():
    def __init__(self):
        return

    def set_exp(self, exp):
        self.exp = exp
        self.initialize()

    def initialize(self):
        return
