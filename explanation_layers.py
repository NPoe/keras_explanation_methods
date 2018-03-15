import keras.backend as K
from keras.layers import Wrapper, LSTM, GRU
import tensorflow as tf
import numpy as np

class ExplanationLayer(Wrapper):
    def __init__(self, layer, **kwargs):
        super(ExplanationLayer, self).__init__(layer, **kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.built = True

class LIME(ExplanationLayer):
    def __init__(self, layer, samples, **kwargs):
        super(LIME, self).__init__(layer, **kwargs)
        self.samples = samples

    def compute_output_shape(self, input_shape):
        output_shape = self.layer.compute_output_shape(input_shape)
        return input_shape[:2] + output_shape[1:]

    def call(self, inputs, mask = None):
        # _sample is implemented by child classes
        # since one instantiation of LIME can handle samples of different sizes,
        # the child must return a list of tensors:
        # [(batchsize, samples, timesteps, ...)]
        samples_by_size, binaries_by_size, masks_by_size = self._sample(inputs, mask)
        minus_one = (-1) * K.ones((1,), dtype = K.dtype(K.shape(inputs)))

        def flatten_01(x):
            return K.reshape(x, K.concatenate([minus_one, K.shape(x)[2:]]))

        samples_by_size_flat = [flatten_01(_x) for _x in samples_by_size]
        # (batchsize * samples, timesteps...)

        if mask:
            masks_by_size_flat = [flatten_01(_m) for _m in masks_by_size]       
            preds_by_size_flat = [self.layer.call(_x, mask = _m)\
                    for _x, _m in zip(samples_by_size_flat, masks_by_size_flat)]
        
        else:
            preds_by_size_flat = [self.layer.call(_x) for _x in samples_by_size_flat]
        
        # [(batchsize * samples, output...)]
        preds_by_size = [K.reshape(_p, K.concatenate([K.shape(inputs)[:1], minus_one, K.shape(_p)[1:]], axis = 0)) \
                for _p in preds_by_size_flat]
        
        preds = K.concatenate(preds_by_size, axis = 1)
        # (batchsize, samples, output_shape)
        
        preds_shape = K.shape(preds)
        preds = K.reshape(preds, K.concatenate([preds_shape[:2], minus_one], axis = 0))
        binary = K.concatenate(binaries_by_size, axis = 1)
        # (batchsize, samples, timesteps)

        def solve(idx):
            squared_matrix = K.dot(binary[idx], K.transpose(binary[idx]))
            inverted = tf.py_func(np.linalg.pinv, [squared_matrix], Tout = K.floatx())
            tmp = K.dot(inverted, binary[idx])
            return K.dot(K.transpose(tmp), preds[idx])

        weights = K.map_fn(lambda idx:solve(idx), K.arange(0, K.shape(inputs)[0]), dtype = K.floatx())
        # (batchsize, timesteps, output_shape_flat)

        weights = K.reshape(weights, K.concatenate([K.shape(weights)[:2], preds_shape[2:]], axis = 0))

        # cut off weights associated with the padded material
        return self._finalize(weights, inputs)

class StandardLIME(LIME):
    def __init__(self, layer, samples, size, axis, **kwargs):
        super(StandardLIME, self).__init_(layer, samples, **kwargs)
        self.axis = axis
        self.size = size

    def _pad(self, x):
        return x

    def _finalize(self, weights, inputs):
        return weights
        
    def _sample(self, inputs, mask=None):
        pattern = [self.axis] + [i if i != self.axis else 0 for i in range(1, K.ndim(inputs))]
        
        inputs_padded = self._pad(inputs) 
        inputs_perm = K.permute_dimensions(inputs_padded, pattern)
        if mask and self.axis <= 1:
            mask_padded = self._pad(mask)
            mask_perm = K.permute_dimensions(mask_padded, pattern)

        batchsize = K.shape(inputs_padded)[0]
        timesteps = K.shape(inputs_padded)[self.axis]
        
        def sample_positions(_):
            return K.random_uniform(shape = (self.size, batchsize), minval=0, maxval=timesteps-self.size, dtype = "int32")

        positions = K.map_fn(get_sample, K.arange(0, self.samples), dtype = "int32")
        # (self.samples, self.size, batchsize)

        def sample_sample(positions):
            return K.permute_dimensions(inputs_perm[positions], pattern)
            
        samples = [K.map_fn(sample_sample, positions, dtype = K.dtype(inputs))]

        def sample_binary(positions):
            tmp = K.sum(K.one_hot(positions, timesteps), axis = 0) # count times every 
            tmp = K.cast(K.cast(tmp, "bool"), "int32") # avoid counts > 0
            return tmp
        
        binaries = [K.map_fn(sample_binary, positions, dtype = K.dtype(inputs))]

        if mask and self.axis <= 1:
            def sample_mask(positions):
                return K.permute_dimensions(mask_perm[positions], pattern)

            masks = [K.map_fn(sample_mask, positions, dtype = K.dtype(mask))]

        else:
            masks = []

        return samples, binaries, masks

        
class LIMSSE(LIME):
    def __init__(self, layer, samples, low=1, high=8, **kwargs):
        super(LIMSSE, self).__init__(layer, samples, **kwargs)
        if low < 1 or high <= low:
            raise Exception("0 < low < high")

        if samples < high - low:
            raise Exception("there must be more samples than high - low")

        self.high = high
        self.low = low

    def _pad(self, x):
        padding_length = K.max(K.stack([self.high - K.shape(x)[1], 0], axis = 0))
        padding_shape = K.concatenate([K.stack([K.shape(x)[0], padding_length], 0), K.shape(x)[2:]], 0)
        return K.concatenate([x, K.zeros(padding_shape)], axis = 1)

    def _finalize(self, weights, inputs):
        return weights[:,:K.shape(inputs)[1]]

    def _sample(self, inputs, mask = None):
        
        substrings_by_size = []
        binaries_by_size = []
        masks_by_size = []
    
        inputs_padded = self._pad(inputs)
        if mask:
            mask_padded = self._pad(mask)
        
        batchsize = K.shape(inputs_padded)[0]
        timesteps = K.shape(inputs_padded)[1]
        
        def sample_length(_):
            return K.random_uniform(shape = (1,), minval=self.low, maxval=self.high, dtype = "int32")
        
        all_lengths = K.arange(self.low, self.high)
        samples_left = self.samples - (self.high - self.low) 
        
        all_lengths = K.concatenate([all_lengths, 
            K.squeeze(K.map_fn(sample_length, K.arange(0, samples_left), dtype = "int32"), axis = -1)], axis = 0)   
        
        for length in range(self.low, self.high):
            times_sampled = K.sum(K.cast(K.equal(all_lengths, length), "int32"))
            
            def sample_start(_):
                return K.random_uniform(shape=(1,), minval=0, maxval=timesteps - length, dtype = "int32")
            starts = K.squeeze(K.map_fn(sample_start, K.arange(times_sampled), dtype = "int32"), axis = -1)
            
            def sample_substring(start):
                return inputs_padded[:,start:start+length]
            
            substrings = K.map_fn(sample_substring, starts, dtype = K.dtype(inputs))
            # (samples, batchsize, length, ...)
            substrings = K.permute_dimensions(substrings, [1,0] + list(range(2, K.ndim(substrings))))
            substrings_by_size.append(substrings)

            def sample_binary(start):
                before = K.zeros((batchsize, start))
                ones = K.ones((batchsize, length))
                after = K.zeros((batchsize, timesteps - start - length))
                return K.concatenate([before, ones, after], axis = 1)
            
            binaries = K.map_fn(sample_binary, starts, dtype = K.floatx())
            # (samples, batchsize, timesteps)
            binaries = K.permute_dimensions(binaries, [1,0,2])
            # (batchsize, samples, timesteps)
            binaries_by_size.append(binaries)

            if mask:
                def sample_submask(start):
                    return mask_padded[:,start:start+length]

                masks = K.map_fn(sample_submask, starts, dtype = K.dtype(mask))
                # (samples, batchsize, length)
                masks = K.permute_dimensions(masks, [1,0,2])
                masks_by_size.append(masks)
        
        return substrings_by_size, binaries_by_size, masks_by_size
        

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
            inputs_ratio = inputs * (alpha / self.intervals)
            output += self._call(inputs_ratio, mask = mask) / self.intervals

        return self._finalize(inputs, output)

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
            
            output += self._call(inputs + noise, mask = mask) / self.samples

        return self._finalize(inputs, output)



class InputPerturbation(ExplanationLayer):
    def __init__(self, layer, size, axis, **kwargs):
        super(InputPerturbation, self).__init__(layer, **kwargs)

        if size < 1:
            raise Exception("size must be > 0")

        self.size = size
        self.supports_masking = True
        self.axis = axis


class InputPerturbation1D(InputPerturbation):
    def __init__(self, layer, size=1, axis=1, **kwargs):
        super(InputPerturbation1D, self).__init__(layer, size, axis, **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + \
                self.layer.compute_output_shape(self._compute_input_shape_int(input_shape))[1:]
   
    def call(self, inputs, mask = None):
       
        shape = K.shape(inputs) # (batch, timesteps, ...)
        int_shape = K.int_shape(inputs)
        timesteps = K.expand_dims(K.expand_dims(K.arange(0, K.shape(inputs)[self.axis]-self.size+1), -1), 0)

        if int_shape[1]:
            def _step(_, states):
                start = states[0]
                _inputs = self._perturb(inputs, start)
                if mask is None:
                    return self.layer.call(_inputs)
                _mask = self._perturb_mask(mask, start)
                return self.layer.call(_inputs, _mask)

       
            _, output, _ = K.rnn(_step, timesteps, initial_states = [0])
        
        else:
        
            def _step(_, states):
                [start, is_mask] = states
                if is_mask:
                    return self._perturb_mask(mask, start), [start+1]
                return self._perturb(inputs, start), [start+1]

            _, _inputs, _ = K.rnn(_step, timesteps, initial_states = [0], constants = [False])

            # batchsize, timesteps - size + 1, _timesteps, [input_shape]
            # omission: _timesteps = timesteps - size
            # occlusion: _timesteps = timesteps

            _input_shape = self._compute_input_shape(shape)
            _reshape = K.concatenate([- K.ones((1,), dtype = K.dtype(_input_shape)), _input_shape[1:]], axis = 0)
            _inputs = K.reshape(_inputs, _reshape)
            # batchsize * (timesteps - size + 1), _timesteps, [input_shape]

            _inputs._uses_learning_phase = inputs._uses_learning_phase
            _inputs._keras_shape = (None, None) + int_shape[2:]

            if not mask is None:
                _, _mask, _ = K.rnn(_step, timesteps, initial_states = [0], constants = [True])
                _mask_reshape = K.concatenate([- K.ones((1,), dtype = K.dtype(_input_shape)), _input_shape[1:2]], axis = 0)
                _mask = K.reshape(_mask, _mask_reshape)
                _output = self.layer.call(_inputs, mask = _mask)
            else:
                _output = self.layer.call(_inputs)
                # (batchsize * (timesteps - size + 1), [output_shape])

            
            output_shape = self.compute_output_shape(int_shape)

            if output_shape[1] is None:
                if output_shape[2] is None:
                    dummy_input = self._perturb(inputs, 0)
                    dummy_output = self.layer.call(dummy_input)
                    return K.reshape(_output, (-1, shape[self.axis]-self.size+1, K.shape(dummy_output)[1]) + output_shape[3:])
                return K.reshape(_output, (-1, shape[self.axis]-self.size+1) + output_shape[2:])
            return K.reshape(_output, (-1,) + output_shape[1:])
    
class InputOmission1D(InputPerturbation1D):
    def __init__(self, layer, size=1, **kwargs):
        super(InputOmission1D, self).__init__(layer, size, axis=1, **kwargs)

    def _perturb_mask(self, mask, start):
        if self.axis > 1:
            return mask
        _mask = mask
        _mask = K.permute_dimensions(_mask, [self.axis, 1-self.axis])
        _mask = K.concatenate([_mask[:start], _mask[start+self.size:]], axis = 0)
        _mask = K.permute_dimensions(_mask, [self.axis, 1-self.axis])
        return _mask

    def _perturb(self, x, start):
        _x = x
        _x = K.permute_dimensions(_x, [self.axis] + [i if i != self.axis else 0 for i in range(1, K.ndim(x))])
        _x = K.concatenate([_x[:start], _x[start+self.size:]], axis = 0)
        _x = K.permute_dimensions(_x, [self.axis] + [i if i != self.axis else 0 for i in range(1, K.ndim(x))])
        
        _shape = K.int_shape(x)
        if _shape[self.axis]:
            _shape = tuple([_shape[i] if i != self.axis else _shape[i] - self.size for i in range(len(_shape))])
        
        _x._keras_shape = _shape
        _x._uses_learning_phase = x._uses_learning_phase
        return _x
    
    def _compute_input_shape_int(self, input_shape):
        if input_shape[self.axis]:
            return tuple([_shape[i] if i != self.axis else _shape[i] - self.size for i in range(len(_shape))])
        return input_shape
    
    def _compute_input_shape(self, input_shape):
        cond = K.equal(K.arange(0, K.shape(input_shape)[0]), self.axis)
        return K.switch(cond, input_shape - self.size, input_shape)

class InputOcclusion1D(InputPerturbation1D):
    def _perturb_mask(self, mask, start):
        return mask

    def _perturb(self, x, start):
        _x = x
        _x = K.permute_dimensions(_x, [self.axis] + [i if i != self.axis else 0 for i in range(1, K.ndim(x))])
        _x = K.concatenate([_x[:start], K.zeros_like(_x[:self.size]), _x[start+self.size:]], axis = 0)
        _x = K.permute_dimensions(_x, [self.axis] + [i if i != self.axis else 0 for i in range(1, K.ndim(x))])

        _x._keras_shape = x._keras_shape
        _x._uses_learning_phase = x._uses_learning_phase
        return _x
    

    def _compute_input_shape_int(self, input_shape):
        return input_shape
    
    def _compute_input_shape(self, input_shape):
        return input_shape



class LRPWrapper(ExplanationLayer):
    def __init__(self, layer, **kwargs):
        super(LRPWrapper, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
        self.built = True

    def call(self, inputs, mask = None):
        outputs = self.layer.call(inputs)
        
        if mask is None:
            return self.layer.lrp(inputs = inputs, outputs = outputs, relevance = outputs, epsilon = self.epsilon)
        return self.layer.lrp(inputs = inputs, outputs = outputs, relevance = outputs, mask = mask, epsilon = self.epsilon)

    def compute_output_shape(self, input_shape):
        output_shape = self.layer.compute_output_shape(input_shape)
        return (output_shape[0]) + input_shape[1:] + output_shape[1:]

def lrp_Dense(self, inputs, linear_outputs, relevance, epsilon = 0.001):

    if not isinstance(self.activation, Linear):
        raise Exception("Activation of Dense layer must be linear in order for LRP to work")

    denominator = epsilon * K.cast(K.greater_equal(linear_outputs, 0), K.floatx()) # (batchsize, out)
    denominator = K.expand_dims(denominator, 1) # (batchsize, 1, out)
    numerator = K.expand_dims(self.kernel, 0) * K.expand_dims(inputs, -1) # (batchsize, in, out)
    relevance = K.expand_dims(relevance, 1) # (batchsize, 1, out)

    return K.sum(relevance * numerator / denominator, axis = -1) # (batchsize, in)

def lrp_Activation(self, inputs, linear_outputs, relevance, epsilon = 0.001):
    return relevance

def lrp_Embedding(self, inputs, linear_outputs, relevance, epsilon = 0.001):
    return K.sum(relevance, axis = 1)

def lrp_GlobalMaxPooling1D(self, inputs, linear_outputs, relevance, epsilon = 0.001):
    equal = K.equals(inputs, K.expand_dims(linear_outputs, 1)) # (batchsize, timesteps, features)
    equal /= K.sum(K.cast(equal, K.floatx()), axis = 1) # in case there were two maxima

    return equal * K.expand_dims(relevance, axis = 1)

def lrp_Conv1D(self, inputs, linear_outputs, relevance, epsilon = 0.001):
    if not isinstance(self.activation, Linear):
        raise Exception("Activation of Conv1D layer must be linear in order for LRP to work")
    
    denominator = epsilon * K.cast(K.greater_equal(linear_outputs, 0), K.floatx())
    denominator = K.expand_dims(denominator, 1)

    def _pad(x, left, right, what, padding):
        pad_left = K.ones_like(x[:,:left]) * what
        pad_right = K.ones_like(x[,:right]) * what
        return K.concatenate([pad_left, x, pad_right], axis = 1)

    if self.padding == "same":
        inputs = _pad(inputs, self.kernel_size[0] // 2, self.kernel_size[0] // 2, 0)

    elif self.padding == "causal":
        inputs = _pad(inputs, self.kernel_size[0] - 1, 0, 0)

    relevance_out = K.expand_dims(relevance, axis = -1)
    relevance_in = 0

    for i in range(self.kernel_size[0]):
        relevance_padded = _pad(relevance_out, i, self.kernel_size[0] - i - 1, 0)
        denominator_padded = _pad(denominator, i, self.kernel_size[0]-i-1, 1)
        numerator = K.expand_dims(inputs, -1) * K.expand_dims(self.kernel[i], 0)
        relevance_in += K.sum(relevance_padded * numerator / denominator_padded, axis = -1)

    if self.padding == "same":
        return relevance_in[:,self.kernel_size[0]//2:-(self.kernel_size[0]//2)]
    elif self.padding == "causal":
        return relevance_in[:,self.kernel_size-1:]

    return relevance_in


def lrp_Sequential(self, inputs, linear_outputs, relevance, mask = None, epsilon = 0.001):
    
    all_inputs, all_masks, all_outputs = [], [], []
    all_masks.append(mask)
    all_inputs.append(inputs)

    for layer in self.layers[:-1]:
        
        if mask is None:
            out = layer.call(all_inputs[-1])
        else:
            out = layer.call(all_inputs[-1], mask = all_masks[-1])

        all_outputs.append(out)
        all_inputs.append(out)
        all_masks.append(layer.compute_mask(all_masks[-1])

    all_outputs.append(linear_outputs)

    for layer in reversed(self.layers):
        mask = all_masks.pop()
        
        if mask is None
            relevance = layer.lrp(inputs = all_inputs.pop(), 
                    outputs = all_outputs.pop(), 
                    relevance = relevance, 
                    epsilon = epsilon)
        else:
            relevance = layer.lrp(inputs = all_inputs.pop(), 
                    outputs = all_outputs.pop(), 
                    relevance = relevance, 
                    epsilon = epsilon,
                    mask = mask)

    return relevance


for layer in ("Dense", "Activation", "Embedding", "Sequential", "Conv1D", "GlobalMaxPooling1D"):
    eval(layer).lrp = eval("lrp_" + layer)


def test():
    import numpy as np
    np.random.seed(123)
    from keras.models import Sequential
    from keras.layers import GlobalMaxPooling1D, Conv1D, Lambda, Embedding, LSTM
    def check_equal(x, y):
        if not x == y:
            print(x)
            print(y)
            print("-------------------")

    def check_close(x,y):
        if not np.allclose(x,y):
            print(x)
            print(y)
            print("-------------------")


    inp = np.array([[1,2,3,3], [1,0,0,0], [1,2,0,0], [3,0,0,0]])
    biginp = np.random.randint(size=(8, 1000), low=0, high=4)

    emb = Embedding(input_dim = 4, output_dim = 9, mask_zero = False)
    embm = Embedding(input_dim = 4, output_dim = 9, mask_zero = True)
    conv = Conv1D(kernel_size = 5, filters = 5, padding="same", input_shape = (None, 9))
    inner = Sequential([conv, GlobalMaxPooling1D()])
    lstm = LSTM(units=5)
    gru = GRU(units=5)

    outer = Sequential([emb, inner])    
    lime = LIMSSE(outer, samples=1000, input_shape = (None,))
    model = Sequential([lime])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))
   
    embmbig = Embedding(input_dim = 4000, output_dim = 200, mask_zero = True)
    grubig = GRU(units=200)
    outerbig = Sequential([embmbig, grubig])    
    limebig = LIMSSE(outerbig, samples=1000, input_shape = (None,))
    modelbig = Sequential([limebig])
    bigout = model.predict_on_batch(biginp)

    outer2 = Sequential([embm, gru])    
    lime = LIMSSE(outer2, samples=10, input_shape = (None,))
    model = Sequential([lime])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))
    bigout = model.predict_on_batch(biginp)

    model = Sequential([emb, InputGradient(layer = inner, input_shape = (None, 9))])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,9,5))

    lam = Lambda(lambda x:3*x, output_shape = lambda x:x)
    model = Sequential([emb, InputGradient(layer = lam)])
    out = model.predict_on_batch(inp)
    for tup in ((1,2), (0,4), (1,2), (0,0)):
        check_equal(out[0,tup[0],tup[1],tup[0],tup[1]], 3.0)
        check_equal(out[1,tup[0],tup[1],tup[0],tup[1]], 3.0)
        check_equal(out[2,tup[0],tup[1],tup[0],tup[1]], 3.0)
        
        check_equal(out[0,tup[0],tup[1],tup[0]+1,tup[1]], 0.0)
        check_equal(out[1,tup[0],tup[1],tup[0],tup[1]+1], 0.0)
        check_equal(out[2,tup[0],tup[1],tup[0]+1,tup[1]], 0.0)

    
    model = Sequential([emb, InputGradientIntegrated(layer = inner, input_shape = (None, 9), intervals = 10)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,9,5))
    
    model = Sequential([emb, InputGradientNoisy(layer = inner, input_shape = (None, 9), noise_ratio = 0.1, samples = 10)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,9,5))
    
    model = Sequential([emb, InputOmission1D(layer = inner, input_shape = (None, 9))])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))
    
    model2 = Sequential([emb, inner])
    check_close(out[:,0], model2.predict_on_batch(inp[:,1:]))

    model = Sequential([emb, InputOcclusion1D(layer = inner, input_shape = (None, 9))])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))
    
    model3 = Sequential([emb])
    model4 = Sequential([inner])
    inp2 = model3.predict_on_batch(inp)
    inp2[:,0] = 0
    check_close(out[:,0], model4.predict_on_batch(inp2))
    
    model = Sequential([emb, InputOmission1D(layer = inner, input_shape = (None, 9))])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))
    
    model = Sequential([emb, InputOcclusion1D(layer = inner, input_shape = (None, 9), size = 2)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,5))
    
    model = Sequential([emb, InputOmission1D(layer = inner, input_shape = (None, 9), size = 2)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,5))
    
    model = Sequential([embm, InputOmission1D(layer = lstm)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))
    
    model = Sequential([embm, InputOcclusion1D(layer = lstm)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))
    
    model = Sequential([embm, InputOcclusion1D(layer = lstm, size = 2)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,5))
    
    model = Sequential([embm, InputOcclusion1D(layer = lstm, axis = 2)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,9,5))
    
    model = Sequential([embm, LSTMDecomposition(layer = lstm, do_forget_gates=False)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))
    
    model = Sequential([embm, LSTMDecomposition(layer = lstm, do_forget_gates=True)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))
    
    model = Sequential([embm, GRUDecomposition(layer = gru, do_forget_gates=False)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))
    
    model = Sequential([embm, GRUDecomposition(layer = gru, do_forget_gates=True)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,4,5))

if __name__ == "__main__":
    test()








