require 'numo/narray'
require './layers_2'
require './numerical_gradient'


class SimpleConvNet
	attr_reader  :params
#    conv - relu - pool - affine - relu - affine - softmax
    
#    Parameters
#    ----------
#    input_size : 入力サイズ（MNISTの場合は784）
#    hidden_size_list : 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
#    output_size : 出力サイズ（MNISTの場合は10）
#    activation : 'relu' or 'sigmoid'
#    weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
#        'relu'または'he'を指定した場合は「Heの初期値」を設定
#        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定

    def initialize(input_dim: [1, 28, 28], 
                 conv_param: {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size: 100, output_size: 10, weight_init_std: 0.01)
        filter_num = conv_param[:filter_num]
        filter_size = conv_param[:filter_size]
        filter_pad = conv_param[:pad]
        filter_stride = conv_param[:stride]
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = (filter_num * (conv_output_size/2) * (conv_output_size/2)).to_i

        # 重みの初期化
        @params = {}
        @params['w1'] = weight_init_std * Numo::DFloat.new(1, filter_num, input_dim[0], filter_size, filter_size).rand_norm
        @params['b1'] = Numo::DFloat.zeros(1, filter_num)
        @params['w2'] = weight_init_std * Numo::DFloat.new(1, pool_output_size, hidden_size).rand_norm
        @params['b2'] = Numo::DFloat.zeros(1, hidden_size)
        @params['w3'] = weight_init_std * Numo::DFloat.new(1, hidden_size, output_size).rand_norm
        @params['b3'] = Numo::DFloat.zeros(1, output_size)

        # レイヤの生成
        @layers = {}
        @layers['Conv1'] = Convolution.new(w: @params['w1'][0, false], b: @params['b1'][0, false], stride: conv_param[:stride], pad: conv_param[:pad])
        @layers['Relu1'] = Relu.new()
        @layers['Pool1'] = Pooling.new(pool_h: 2, pool_w: 2, stride: 2)
        @layers['Affine1'] = Affine.new(w: @params['w2'][0, false], b: @params['b2'][0, false])
        @layers['Relu2'] = Relu.new()
        @layers['Affine2'] = Affine.new(w: @params['w3'][0, false], b: @params['b3'][0, false])

        @last_layer = SoftmaxWithLoss.new()
	end

    def predict(x:, train_flag:false)
 	   	@layers.values.inject(x) {|x, layer| x = layer.forward(x:x, train_flag:train_flag) }
    end

    def loss(x:, t:, train_flag:false)	#損失関数を求める
	    y = predict(x:x, train_flag:train_flag)
        return @last_layer.forward(x:y, t:t)
    end

    def accuracy(x:, t:, batch_size: 100)
        t = t.max_index(1) % t.shape[1]  if t.ndim != 1
        
        acc = 0.0
        
        (x.shape[0] / batch_size).to_i.times{|i|
        	tx = x[i*batch_size...(i+1)*batch_size, false]
        	tt = t[i*batch_size...(i+1)*batch_size, false]
        	y = predict(x:tx)
        	y = y.max_index(1) % y.shape[1]
            acc += y.eq(tt).cast_to(Numo::UInt16).sum
        }
        return acc / x.shape[0]
	end

    def numerical_gradients(x:, t:) # 勾配を求める（数値微分）
        loss_w = lambda { loss(x:x, t:t, train_flag:true) }
        grads = {}
        [1, 2, 3].each{|idx|
	        grads["w#{idx}"] = numerical_gradient(loss_w, @params["w#{idx}"][0, false])
    	    grads["b#{idx}"] = numerical_gradient(loss_w, @params["b#{idx}"][0, false])
        }
        return grads
	end

    def gradient(x:, t:)	# 勾配を求める（誤差逆伝搬法）
        # forward
 	    loss(x: x, t: t, train_flag: true)

        # backward
        dout = 1
        dout = @last_layer.backward(dout: dout)

	    layers = @layers.values.reverse
    	layers.inject(dout) {|dout, layer|
	        dout = layer.backward(dout: dout)
    	}

        # 設定
        grads = {}
        grads['w1'], grads['b1'] = @layers['Conv1'].dw, @layers['Conv1'].db
        grads['w2'], grads['b2'] = @layers['Affine1'].dw, @layers['Affine1'].db
        grads['w3'], grads['b3'] = @layers['Affine2'].dw, @layers['Affine2'].db

        return grads
    end
=begin        
    def save_params(file_name: "params.pkl")
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
	end
    def load_params(file_name: "params.pkl")
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
    end
=end
end
