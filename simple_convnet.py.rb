require 'numo/narray'
require './layers_2'
require './numerical_gradient'


class SimpleConvNet
#    conv - relu - pool - affine - relu - affine - softmax
    
#    Parameters
#    ----------
#    input_size : ���̓T�C�Y�iMNIST�̏ꍇ��784�j
#    hidden_size_list : �B��w�̃j���[�����̐��̃��X�g�ie.g. [100, 100, 100]�j
#    output_size : �o�̓T�C�Y�iMNIST�̏ꍇ��10�j
#    activation : 'relu' or 'sigmoid'
#    weight_init_std : �d�݂̕W���΍����w��ie.g. 0.01�j
#        'relu'�܂���'he'���w�肵���ꍇ�́uHe�̏����l�v��ݒ�
#        'sigmoid'�܂���'xavier'���w�肵���ꍇ�́uXavier�̏����l�v��ݒ�

    def initialize(input_dim: [1, 28, 28], 
                 conv_param: {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size: 100, output_size: 10, weight_init_std: 0.01)
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = (filter_num * (conv_output_size/2) * (conv_output_size/2)).to_i

        # �d�݂̏�����
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # ���C���̐���
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(x:, train_flag: false)
 	   	@layers.values.inject(x) do |x, layer|
     		x = layer.forward(x: x, train_flag: train_flag)
		end
        return x
    end

    def loss(x:, t:, train_flag: false)	#�����֐������߂�
	    y = predict(x: x, train_flag: train_flag)
        return @last_layer.forward(x: y, t: t)
    end

    def accuracy(x:, t:, batch_size: 100)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]
	end

    def numerical_gradients(x:, t:) # ���z�����߂�i���l�����j
        loss_w = lambda { loss(x: x, t: t, train_flag: true) }
        grads = {}
        [1, 2, 3].each{|idx|
	        grads["w#{idx}"] = numerical_gradient(loss_w, @params["w#{idx}"][0, false])
    	    grads["b#{idx}"] = numerical_gradient(loss_w, @params["b#{idx}"][0, false])
        }
        return grads
	end

    def gradient(x:, t:)	# ���z�����߂�i�덷�t�`���@�j
        # forward
 	    loss(x: x, t: t, train_flag: true)

        # backward
        dout = 1
        dout = @last_layer.backward(dout: dout)

	    layers = @layers.values.reverse
    	layers.inject(dout) {|dout, layer|
	      dout = layer.backward(dout: dout)
    	}

        # �ݒ�
        grads = {}
        grads['W1'], grads['b1'] = @layers['Conv1'].dW, @layers['Conv1'].db
        grads['W2'], grads['b2'] = @layers['Affine1'].dW, @layers['Affine1'].db
        grads['W3'], grads['b3'] = @layers['Affine2'].dW, @layers['Affine2'].db

        return grads
    end
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
            
end
