def test():
    
    import numpy as np
    np.random.seed(123)

    from keras.models import Sequential
    from keras.layers import GlobalMaxPooling1D, Conv1D, Lambda, Embedding, LSTM, GRU

    from lime_layers import LIMSSE, StandardLIME
    from gradient_layers import InputGradient, InputGradientNoisy, InputGradientIntegrated
    from perturbation_layers import InputOmission1D, InputOcclusion1D

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


    inp = np.array([[1,2,3], [1,0,0], [0,2,1], [3,0,0]])

    HIDDEN_SIZE = 5
    EMB_SIZE = 9

    emb = Embedding(input_dim = inp.max()+1, output_dim = EMB_SIZE, mask_zero = False)
    embm = Embedding(input_dim = inp.max()+1, output_dim = EMB_SIZE, mask_zero = True)
    conv = Conv1D(kernel_size = HIDDEN_SIZE, filters = HIDDEN_SIZE, padding="same", input_shape = (None, EMB_SIZE))
    inner = Sequential([conv, GlobalMaxPooling1D()])
    lstm = LSTM(units=5, input_shape = (None, EMB_SIZE))
    gru = GRU(units=5, input_shape = (None, EMB_SIZE))

    outer2 = Sequential([embm, gru])    
    lime = LIMSSE(outer2, samples=20, input_shape = (None,))
    model = Sequential([lime])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,HIDDEN_SIZE))
    tmp1 = abs(out).sum(axis = -1) > 0.01
    tmp2 = inp > 0.0
    # not a real test, but check that masked values from inp (0) have relevances close to zero
    # if they do, assume this works
    check_close(tmp1, tmp2)
    
    lime = StandardLIME(outer2, samples=50, axis=1, size=2, input_shape = (None,))
    model = Sequential([lime])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,HIDDEN_SIZE))
    tmp1 = abs(out).sum(axis = -1) > 0.01
    tmp2 = inp > 0.0
    check_close(tmp1, tmp2)

    model = Sequential([emb, InputGradient(layer = inner, input_shape = (None, 9))])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,EMB_SIZE,HIDDEN_SIZE))

    # check input gradient with easy derivative
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

    
    model = Sequential([emb, InputGradientIntegrated(layer = inner, input_shape = (None, EMB_SIZE), intervals = 10)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,EMB_SIZE,HIDDEN_SIZE))
    
    model = Sequential([emb, InputGradientNoisy(layer = inner, input_shape = (None, EMB_SIZE), noise_ratio = 0.1, samples = 10)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,EMB_SIZE,HIDDEN_SIZE))
    
    
    inner_lstm = Sequential([lstm])
    model = Sequential([emb, InputOcclusion1D(layer = inner_lstm, input_shape = (None, EMB_SIZE), axis = 1, size = 1)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,HIDDEN_SIZE))
    
    model3 = Sequential([emb])
    model4 = Sequential([inner_lstm])
    inp2 = model3.predict_on_batch(inp)
    inp2[:,0] = 0
    check_close(out[:,0], model4.predict_on_batch(inp2))
    
    model = Sequential([emb, InputOcclusion1D(layer = inner_lstm, input_shape = (None, EMB_SIZE), axis = 2, size = 1)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,EMB_SIZE,HIDDEN_SIZE))
    
    model3 = Sequential([emb])
    model4 = Sequential([inner_lstm])
    inp2 = model3.predict_on_batch(inp)
    inp2[:,:,0] = 0
    check_close(out[:,0], model4.predict_on_batch(inp2))

    model = Sequential([emb, InputOmission1D(layer = inner, input_shape = (None, EMB_SIZE))])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,HIDDEN_SIZE))
    
    model2 = Sequential([emb, inner])
    check_close(out[:,0], model2.predict_on_batch(inp[:,1:]))

    model = Sequential([emb, InputOcclusion1D(layer = inner, input_shape = (None, EMB_SIZE), axis = 1, size = 1)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,HIDDEN_SIZE))
    
    model3 = Sequential([emb])
    model4 = Sequential([inner])
    inp2 = model3.predict_on_batch(inp)
    inp2[:,0] = 0
    check_close(out[:,0], model4.predict_on_batch(inp2))
    
    model = Sequential([emb, InputOmission1D(layer = inner, input_shape = (None, EMB_SIZE))])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,HIDDEN_SIZE))
    
    model = Sequential([emb, InputOcclusion1D(layer = inner, input_shape = (None, EMB_SIZE), size = 2)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,2,HIDDEN_SIZE))
    
    model = Sequential([emb, InputOmission1D(layer = inner, input_shape = (None, EMB_SIZE), size = 2)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,2,HIDDEN_SIZE))
    
    model = Sequential([embm, InputOmission1D(layer = lstm)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,HIDDEN_SIZE))
    
    model = Sequential([embm, InputOcclusion1D(layer = lstm)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,3,HIDDEN_SIZE))
    
    model = Sequential([embm, InputOcclusion1D(layer = lstm, size = 2)])
    out = model.predict_on_batch(inp)
    check_equal(out.shape, (4,2,HIDDEN_SIZE))
    

    print("Checks done")

if __name__ == "__main__":
    test()








