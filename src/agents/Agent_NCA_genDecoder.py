from src.agents.Agent_NCA_genImage import Agent_NCA_genImage

class Agent_NCA_gen_Decoder(Agent_NCA_genImage):
    def get_outputs(self, data, full_img=False, **kwargs):
        r"""Get the outputs of the model
            #Args
                data (int, tensor, tensor): id, inputs, targets
        """
        id, inputs, targets, vec = data['id'], data['image'], data['label'], data['image_vec']
        outputs = self.model(targets, vec, steps=self.getInferenceSteps(), fire_rate=self.exp.get_from_config('cell_fire_rate'))

        return outputs, targets