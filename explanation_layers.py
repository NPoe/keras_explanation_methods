from keras.layers import Wrapper

class ExplanationLayer(Wrapper):
    def __init__(self, layer, **kwargs):
        super(ExplanationLayer, self).__init__(layer, **kwargs)
        if hasattr(self.layer, "supports_masking") and self.layer.supports_masking:
            self.supports_masking = True
        else:
            self.supports_masking = False

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.built = True
