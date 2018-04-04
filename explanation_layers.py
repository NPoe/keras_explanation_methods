from keras.layers import Wrapper

class ExplanationLayer(Wrapper):
    def __init__(self, layer, **kwargs):
        super(ExplanationLayer, self).__init__(layer, **kwargs)
        self.supports_masking = self.layer.supports_masking

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.built = True
