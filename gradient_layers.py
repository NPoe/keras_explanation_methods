import keras.backend as K
from explanation_layers import ExplanationLayer

class InputGradient(ExplanationLayer):
    """Gradients of output w.r.t. input -- Simonyan et al. 2013

    This layer wraps around a layer L with input shape I (ignoring batch size) and output 
    shape O (ignoring batch size). It returns a tensor of shape (batch size,) | I | O, 
    where every element is a partial derivative of an output with respect to an input.
    For instance: I (batch_size, timesteps, embedding_dim) and O (batch_size, classes)
    gives gradient output shape (batch_size, timesteps, embedding_dim, classes).
    
    # Arguments
        layer: wrapped layer
    
    # Known issues: 
        The gradient layer does not work on RNNs that are not unrolled, due to an issue
        with nested while loops in tensorflow. Even when RNNs are unrolled, the gradient
        propagation is very slow. In this case, consider a Lambda layer to select
        the relevant output before applying the gradients.
    """

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
    """Integrated gradients of output w.r.t. input -- Sundararajan et al. 2017.
    
    The integrated gradient layer calculates gradients on an interpolation ``path''
    between an all-zero input and the actual input. This helps against the small
    gradient problem in cases where inputs are saturating a non-linearity.

    # Arguments
        layer: wrapped layer
        intervals: number of steps on the interpolation path. The larger this value,
        the closer the result will be to the true integral.
    
    # Known issues: 
        See InputGradient layer.
    """

    def __init__(self, layer, intervals = 20, **kwargs):
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
    """Noisy gradients of output w.r.t. input -- Smilkov et al. 2017.
    
    The noisy gradient layer adds Gaussian noise to the input and then calculates
    the average gradient of the output with respect to the noisy input.
    This is supposed to help against small-scale fluctuations in the space
    immediately surrounding the input.

    # Arguments
        layer: wrapped layer
        samples: number of samples to draw for a single input
        noise_ratio: the std of the Gaussian noise is calculated as
        (noise_ratio * [max(input) - min(input)])
    
    # Known issues: 
        See InputGradient layer.
    """
    def __init__(self, layer, noise_ratio = 0.1, samples = 20, **kwargs):
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
