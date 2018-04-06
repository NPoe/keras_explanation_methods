import keras.backend as K
from explanation_layers import ExplanationLayer

class InputPerturbation(ExplanationLayer):
    def __init__(self, layer, size = 1, axis = 1, **kwargs):
        super(InputPerturbation, self).__init__(layer, **kwargs)

        if size < 1:
            raise Exception("size must be > 0")

        self.size = size
        self.axis = axis

class InputPerturbation1D(InputPerturbation):
    def compute_output_shape(self, input_shape):
        if input_shape[self.axis] == None:
            x = None
        else:
            x = input_shape[self.axis] - self.size + 1
        return input_shape[:1] + (x,) + \
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

            if self.axis <= 1 and not mask is None:
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
        pattern = [self.axis] + [i if i != self.axis else 0 for i in range(1, K.ndim(x))]
        _x = K.permute_dimensions(_x, pattern)
        _x = K.concatenate([_x[:start], _x[start+self.size:]], axis = 0)
        _x = K.permute_dimensions(_x, pattern)
        
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
        pattern = [self.axis] + [i if i != self.axis else 0 for i in range(1, K.ndim(x))]
        _x = K.permute_dimensions(_x, pattern)
        _x = K.concatenate([_x[:start], K.zeros_like(_x[:self.size]), _x[start+self.size:]], axis = 0)
        _x = K.permute_dimensions(_x, pattern)

        _x._keras_shape = x._keras_shape
        _x._uses_learning_phase = x._uses_learning_phase
        return _x
    

    def _compute_input_shape_int(self, input_shape):
        return input_shape
    
    def _compute_input_shape(self, input_shape):
        return input_shape
