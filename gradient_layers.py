import keras.backend as K
from explanation_layer import ExplanationLayer

class InputGradient(ExplanationLayer):
    def __init__(self, layer, **kwargs):
        super(InputGradient, self).__init__(layer, **kwargs)

    def _call(self, inputs, mask = None):
        if mask is None:
            self.layer_output = self.layer.call(inputs)
        else:
            self.layer_output = self.layer.call(inputs, mask = mask)

        layer_output_flat = K.flatten(self.layer_output)

        num_outputs = K.shape(layer_output_flat)[0]
        return K.map_fn(lambda i:K.gradients(layer_output_flat[i], inputs)[0], 
                K.arange(0, num_outputs), dtype = K.dtype(layer_output_flat))

    def _finalize(self, inputs, output):
        # (all output dims multiplied, batchsize, remaining input dims)

        output = K.permute_dimensions(output, [1] + list(range(2, K.ndim(output))) + [0])
        # (batchsize, [remaining_input_dims], all_output_dims multiplied)
        
        output = K.reshape(output, K.concatenate([K.shape(inputs), K.shape(self.layer_output)]))
        # (batchsize, [remaining_input_dims], batchsize, [remaining_output_dims])

        dim_in = K.ndim(inputs)
        dim_out = len(self.layer.compute_output_shape(K.int_shape(inputs)))

        output = K.permute_dimensions(output, [0] + [dim_in] + list(range(1, dim_in)) + list(range(dim_in+1, dim_in+dim_out)))
       
        batchsize = K.shape(inputs)[0]
        output = K.map_fn(lambda i:output[i,i], K.arange(0, batchsize), dtype = K.dtype(output))
        
        return output

    def call(self, inputs, mask = None):

        output = self._call(inputs, mask = mask)
        return self._finalize(inputs, output = output)

    def compute_output_shape(self, input_shape):
        return input_shape + self.layer.compute_output_shape(input_shape)[1:]


class InputGradientIntegrated(InputGradient):
    def __init__(self, layer, intervals, **kwargs):
        super(InputGradientIntegrated, self).__init__(layer, **kwargs)
        if intervals < 1:
            raise Exception("intervals must be > 0")
        
        self.intervals = intervals

    def call(self, inputs, mask = None):
        output = 0
        for alpha in range(self.intervals):
            output += self._call(inputs * (alpha / self.intervals), mask = mask)

        return self._finalize(inputs, output / self.intervals)

class InputGradientNoisy(InputGradient):
    def __init__(self, layer, noise_ratio, samples, **kwargs):
        super(InputGradientNoisy, self).__init__(layer, **kwargs)
        self.noise_ratio = noise_ratio
        self.samples = samples

    def call(self, inputs, mask = None):
        output = 0
        for _ in range(self.samples):

            tmpmax = self.noise_ratio * inputs
            tmpmin = self.noise_ratio * inputs
            for i in range(K.ndim(inputs) - 1):
                tmpmax = K.max(tmpmax, axis = -1)
                tmpmin = K.min(tmpmin, axis = -1)
            tmpdiff = tmpmax - tmpmin

            batchshape = K.shape(inputs)[1:]
            batchsize = K.shape(inputs)[0]
            noise = K.map_fn(lambda i:K.random_normal(batchshape, stddev = tmpdiff[i]),
                    K.arange(0, batchsize), dtype = K.dtype(inputs))
            
            output += self._call(inputs + noise, mask = mask)

        return self._finalize(inputs, output / self.samples)
