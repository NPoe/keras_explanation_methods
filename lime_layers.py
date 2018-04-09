import keras.backend as K
import tensorflow as tf
import numpy as np
from explanation_layers import ExplanationLayer

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
            tmp = K.reshape(x, K.concatenate([minus_one, K.shape(x)[2:]]))
            shape = K.int_shape(x)
            if shape[0] and shape[1]:
                tmp._keras_shape = (shape[0] * shape[1],) + shape[2:]
            else:
                tmp._keras_shape = (None,) + shape[2:]
            tmp._uses_learning_phase = x._uses_learning_phase
            return tmp

        samples_by_size_flat = [flatten_01(_x) for _x in samples_by_size]
        # (batchsize * samples, timesteps...)
        

        if not mask is None:
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
    def __init__(self, layer, samples, axis, size = 6, **kwargs):
        super(StandardLIME, self).__init__(layer, samples, **kwargs)
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
        int_shape = K.int_shape(inputs)

        if mask and self.axis <= 1:
            mask_padded = self._pad(mask)
            mask_perm = K.permute_dimensions(mask_padded, pattern)

        batchsize = K.shape(inputs_padded)[0]
        timesteps = K.shape(inputs_padded)[self.axis]
        
        def sample_positions(_):
            return K.random_uniform(shape = (self.size,), minval=0, maxval=timesteps, dtype = "int32")

        positions = K.map_fn(sample_positions, K.arange(0, self.samples), dtype = "int32")

        def sample_sample(pos):
            tmp = K.gather(inputs_perm, pos)
            return K.permute_dimensions(tmp, pattern)
           
        samples = K.map_fn(sample_sample, positions, dtype = K.dtype(inputs))
        samples = [K.permute_dimensions(samples, [1,0] + list(range(2, K.ndim(samples))))]
        
        samples[0]._keras_shape = (None, None, self.size) + int_shape[2:]
        samples[0]._uses_learning_phase = inputs._uses_learning_phase

        def sample_binary(positions):
            tmp = K.one_hot(positions, timesteps)
            tmp = K.sum(tmp, axis = 0)
            tmp = K.cast(K.cast(tmp, "bool"), K.dtype(inputs)) # avoid counts > 1
            tmp = K.expand_dims(tmp, 0) # reintroduce batch size
            tmp = K.tile(tmp, (batchsize,) + tuple([1] * (K.ndim(tmp)-1)))
            # (batchsize, timesteps) 
            return tmp
        
        binaries = K.map_fn(sample_binary, positions, dtype = K.dtype(inputs))
        # (samples, batchsize, timesteps)
        binaries = [K.permute_dimensions(binaries, [1,0,2])]
        # (batchsize, samples, timesteps)

        if mask and self.axis <= 1:
            def sample_mask(positions):
                return K.permute_dimensions(mask_perm[positions], pattern)

            masks = K.map_fn(sample_mask, positions, dtype = K.dtype(mask))
            masks = [K.permute_dimensions(masks, [1,0,2])]

        else:
            masks = []

        return samples, binaries, masks

        
class LIMSSE(LIME):
    def __init__(self, layer, samples, low=2, high=8, **kwargs):
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
        return K.concatenate([x, K.zeros(padding_shape, dtype = K.dtype(x))], axis = 1)

    def _finalize(self, weights, inputs):
        return weights[:,:K.shape(inputs)[1]]

    def _sample(self, inputs, mask = None):
        
        substrings_by_size = []
        binaries_by_size = []
        masks_by_size = []
    
        inputs_padded = self._pad(inputs)
        if not mask is None:
            mask_padded = self._pad(mask)
        
        batchsize = K.shape(inputs_padded)[0]
        timesteps = K.shape(inputs_padded)[1]
        int_shape = K.int_shape(inputs)
        
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
            substrings._keras_shape = (None, None, length) + int_shape[2:]
            substrings._uses_learning_phase = inputs._uses_learning_phase
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

            if not mask is None:
                def sample_submask(start):
                    return mask_padded[:,start:start+length]

                masks = K.map_fn(sample_submask, starts, dtype = K.dtype(mask))
                # (samples, batchsize, length)
                masks = K.permute_dimensions(masks, [1,0,2])
                masks_by_size.append(masks)
        
        return substrings_by_size, binaries_by_size, masks_by_size
