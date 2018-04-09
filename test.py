import numpy as np
np.random.seed(12345)

from keras.models import Sequential, Model
from keras.layers import *

from lime_layers import *
from gradient_layers import *
from perturbation_layers import *

import unittest

class Test_on_Model(unittest.TestCase):
    """
    This test case trains a simple model to return True when an input contains the symbol
    TARGET and False otherwise. Tests check whether explanation methods subsequently
    place maximum relevance on TARGET.
    """
    
    TARGET = 25
    
    @classmethod
    def setUpClass(cls):
        cls.train_model()
    
    def setUp(self):
        self.make_inputs()

    @classmethod
    def train_model(cls):
        def generator():
            while True:
                X = np.random.randint(cls.TARGET*2, size = (8, cls.TARGET))
                Y = (X == cls.TARGET).sum(axis = -1, keepdims = True).clip(0,1)
                yield(X, Y)
    
        gen = generator()
            
        cls.emb = Embedding(input_dim = cls.TARGET*2, output_dim = 5)
        cls.conv = Conv1D(filters = 5, kernel_size = 3, padding="same", input_shape = (None, 5))
        cls.pool = GlobalMaxPooling1D()
        cls.dense = Dense(units = 1, activation = "sigmoid")
        cls.primary = Sequential([cls.emb, cls.conv, cls.pool, cls.dense])
        cls.primary.compile(loss="binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
        cls.primary.fit_generator(gen, epochs = 1, steps_per_epoch = 10000, verbose = 0)
        
        cls.primary_no_emb = Sequential([cls.conv, cls.pool, cls.dense])
        cls.primary_emb_only = Sequential([cls.emb])
        cls.trained = True
        
        cls.l1norm = Lambda(lambda x: K.sum(K.abs(x), axis = 2))
        cls.l2norm = Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis = 2)))

        cls.inp = Input((None,))
        cls.dot = Dot(2)
        
        cls.simple_gradient = InputGradient(cls.primary_no_emb, input_shape = (None,))
        cls.integrated_gradient = InputGradientIntegrated(cls.primary_no_emb, intervals = 10, input_shape = (None,))
        cls.noisy_gradient = InputGradientNoisy(cls.primary_no_emb, samples = 10, noise_ratio = 0.1, input_shape = (None,))

    def make_inputs(self):
        inputs = np.arange(2*self.TARGET)
        np.random.shuffle(inputs)
        self.position = np.where(inputs==self.TARGET)[0]
        self.inputs = np.expand_dims(inputs, 0)
   
    def _test_expmodel(self, expmodel):
        prediction = expmodel.predict(self.inputs)[0]
        self.assertEqual(prediction.argmax(), self.position)

    def test_limsse_on_model(self):
        limsse = LIMSSE(self.primary, samples = 100, input_shape = (None,))
        self._test_expmodel(Sequential([limsse]))

    def test_lime_on_model(self):
        lime = StandardLIME(self.primary, samples = 100, axis = 1, input_shape = (None,))
        self._test_expmodel(Sequential([lime]))
    
    def test_omit_on_model(self):
        omit = InputOmission1D(self.primary_no_emb, mode = "minus", size = 1, input_shape = (None,))
        self._test_expmodel(Sequential([self.emb, omit]))

    def test_occ_on_model(self):
        occ = InputOcclusion1D(self.primary_no_emb, mode = "minus", size = 1, axis = 1, input_shape = (None,))
        self._test_expmodel(Sequential([self.emb, occ]))

    def test_grad_simple_l1_on_model(self):
        model = Sequential([self.emb, self.simple_gradient, self.l1norm])
        self._test_expmodel(model)
    
    def test_grad_integ_l1_on_model(self):
        model = Sequential([self.emb, self.integrated_gradient, self.l1norm])
        self._test_expmodel(model)
    
    def test_grad_noisy_l1_on_model(self):
        model = Sequential([self.emb, self.noisy_gradient, self.l1norm])
        self._test_expmodel(model)
    
    def test_grad_simple_l2_on_model(self):
        model = Sequential([self.emb, self.simple_gradient, self.l2norm])
        self._test_expmodel(model)
    
    def test_grad_integ_l2_on_model(self):
        model = Sequential([self.emb, self.integrated_gradient, self.l2norm])
        self._test_expmodel(model)
    
    def test_grad_noisy_l2_on_model(self):
        model = Sequential([self.emb, self.noisy_gradient, self.l2norm])
        self._test_expmodel(model)
    
    def test_grad_simple_dot_on_model(self):
        model = Model([self.inp], [self.dot([self.emb(self.inp), self.simple_gradient(self.emb(self.inp))])])
        self._test_expmodel(model)
    
    def test_grad_integ_dot_on_model(self):
        model = Model([self.inp], [self.dot([self.emb(self.inp), self.integrated_gradient(self.emb(self.inp))])])
        self._test_expmodel(model)
    
    def test_grad_noisy_dot_on_model(self):
        model = Model([self.inp], [self.dot([self.emb(self.inp), self.noisy_gradient(self.emb(self.inp))])])
        self._test_expmodel(model)


class Test_Unittests(unittest.TestCase):
    """
    Output shape tests as well as miscellaneous unit tests.
    """
    
    EMB_SIZE = 9
    HIDDEN_SIZE = 3

    @classmethod
    def setUpClass(cls):
        cls.input = np.array([[1,2,3], [1,0,0], [0,2,1], [3,0,0]])
    
        cls.BATCH_SIZE = cls.input.shape[0]
        cls.TIMESTEPS = cls.input.shape[1]

        cls.emb = Embedding(input_dim = cls.input.max()+1, output_dim = cls.EMB_SIZE, mask_zero = False, input_shape = (None,))
        cls.emb_masking = Embedding(input_dim = cls.input.max()+1, output_dim = cls.EMB_SIZE, mask_zero = True, input_shape = (None,))
    
        cls.conv = Conv1D(kernel_size = cls.HIDDEN_SIZE, filters = cls.HIDDEN_SIZE, padding="same", input_shape = (None, cls.EMB_SIZE))
        cls.pool = GlobalMaxPooling1D()

        cls.conv_pool_seq = Sequential([cls.conv, cls.pool])
    
        cls.lstm = LSTM(units = cls.HIDDEN_SIZE, input_shape = (None, cls.EMB_SIZE))
        cls.gru = GRU(units = cls.HIDDEN_SIZE, input_shape = (None, cls.EMB_SIZE))
        
        cls.lstm_seq = Sequential([cls.lstm])
        cls.gru_seq = Sequential([cls.gru])

    def _test_shape(self, expmodel, shape):
        out = expmodel.predict(self.input)
        self.assertEqual(out.shape, shape)

    def test_limsse_shape(self):
        for layer in (self.lstm, self.gru, self.conv_pool_seq):
            for emb in (self.emb, self.emb_masking):
                # keras conv1D does not support masking
                if emb == self.emb_masking and layer == self.conv_pool_seq:
                    continue

                inner_model = Sequential([emb, layer])
                limsse = LIMSSE(inner_model, samples=20, input_shape = (None,))
                self._test_shape(Sequential([limsse]), (self.BATCH_SIZE, self.TIMESTEPS, self.HIDDEN_SIZE))
    
    def test_lime_shape(self):
        for layer in (self.lstm, self.gru, self.conv_pool_seq):
            for emb in (self.emb, self.emb_masking):
                # keras conv1D does not support masking
                if emb == self.emb_masking and layer == self.conv_pool_seq:
                    continue
                inner_model = Sequential([emb, layer])
                lime = StandardLIME(inner_model, samples=20, axis = 1, input_shape = (None,))
                self._test_shape(Sequential([lime]), (self.BATCH_SIZE, self.TIMESTEPS, self.HIDDEN_SIZE))
    
    def test_omit_shape(self):
        for layer in (self.lstm, self.gru, self.conv_pool_seq):
            for emb in (self.emb, self.emb_masking):
                # keras conv1D does not support masking
                if emb == self.emb_masking and layer == self.conv_pool_seq:
                    continue
                for size in (1,2):
                    omit = InputOmission1D(layer, size = size, input_shape = (None,))
                    self._test_shape(Sequential([emb, omit]), (self.BATCH_SIZE, self.TIMESTEPS - size + 1, self.HIDDEN_SIZE))
    
    def test_occ_shape(self):
        for layer in (self.lstm, self.gru, self.conv_pool_seq):
            for emb in (self.emb, self.emb_masking):
                # keras conv1D does not support masking
                if emb == self.emb_masking and layer == self.conv_pool_seq:
                    continue
                for size in (1,2):
                    occ1 = InputOcclusion1D(layer, size = size, axis=1, input_shape = (None,))
                    self._test_shape(Sequential([emb, occ1]), (self.BATCH_SIZE, self.TIMESTEPS - size + 1, self.HIDDEN_SIZE))
                    occ2 = InputOcclusion1D(layer, size = size, axis=2, input_shape = (None,))
                    self._test_shape(Sequential([emb, occ2]), (self.BATCH_SIZE, self.EMB_SIZE - size + 1, self.HIDDEN_SIZE))
    
    def test_grad_shape(self):
        layer = self.conv_pool_seq
        simple_gradient = InputGradient(layer)
        self._test_shape(Sequential([self.emb, simple_gradient]), (self.BATCH_SIZE, self.TIMESTEPS, self.EMB_SIZE, self.HIDDEN_SIZE))
        integrated_gradient = InputGradientIntegrated(layer, intervals = 20)
        self._test_shape(Sequential([self.emb, integrated_gradient]), (self.BATCH_SIZE, self.TIMESTEPS, self.EMB_SIZE, self.HIDDEN_SIZE))
        noisy_gradient = InputGradientNoisy(layer, noise_ratio = 0.1, samples = 20)
        self._test_shape(Sequential([self.emb, noisy_gradient]), (self.BATCH_SIZE, self.TIMESTEPS, self.EMB_SIZE, self.HIDDEN_SIZE))
                
    def test_grad(self):
        triple_layer = Lambda(lambda x:3*x, output_shape = lambda shape:shape)

        simple_gradient = InputGradient(layer = triple_layer)
        integrated_gradient = InputGradientIntegrated(layer = triple_layer, intervals = 10)
        noisy_gradient = InputGradientNoisy(layer = triple_layer, samples = 10, noise_ratio = 0.1)

        for grad_layer in (simple_gradient, integrated_gradient, noisy_gradient):
            model = Sequential([self.emb, grad_layer])
            out = model.predict_on_batch(self.input)
            for tup in ((1,2), (0,4), (1,2), (0,0)):
                self.assertEqual(out[0,tup[0],tup[1],tup[0],tup[1]], 3.0)
                self.assertEqual(out[2,tup[0],tup[1],tup[0],tup[1]], 3.0)
        
                self.assertEqual(out[0,tup[0],tup[1],tup[0]+1,tup[1]], 0.0)
                self.assertEqual(out[2,tup[0],tup[1],tup[0]+1,tup[1]], 0.0)
    
    def test_omit(self):
        for layer in (self.lstm_seq, self.gru_seq, self.conv_pool_seq,):
            for size in (1,2):
                model = Sequential([self.emb, InputOmission1D(layer, size = size)])
                out = model.predict(self.input)

                omit_layer = Lambda(lambda x:x[:,size:], 
                        output_shape = lambda shape:shape if shape[1] is None else (shape[0], shape[1]-size+1) + shape[2:])

                model_with = Sequential([self.emb, omit_layer, layer])
                model_without = Sequential([self.emb, layer])
                
                orig_out = model_without.predict(self.input)
                perturbed_out = model_with.predict(self.input)
                self.assertTrue(np.allclose(out[:,0], orig_out - perturbed_out))
    
    def test_occ(self):
        for layer in (self.lstm_seq, self.gru_seq, self.conv_pool_seq,):
            for size in (1,2):
                for axis in (1,2):
                    model = Sequential([self.emb, InputOcclusion1D(layer, size = size, axis = axis)])
                    out = model.predict(self.input)
                
                    if axis == 1:
                        occ_layer = Lambda(lambda x:K.concatenate([x[:,:size] * 0, x[:,size:]], axis = 1), 
                                output_shape = lambda shape:shape)
                    else:
                        occ_layer = Lambda(lambda x:K.concatenate([x[:,:,:size] * 0, x[:,:,size:]], axis = 2), 
                                output_shape = lambda shape:shape)

                    model_with = Sequential([self.emb, occ_layer, layer])
                    model_without = Sequential([self.emb, layer])
                
                    orig_out = model_without.predict(self.input)
                    perturbed_out = model_with.predict(self.input)
                    self.assertTrue(np.allclose(out[:,0], orig_out - perturbed_out))



if __name__ == "__main__":
    unittest.main()
