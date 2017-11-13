require 'numo/narray'
require './layers_2'

class DeepConvNet
	attr_reader  :params
    # 認識率99%以上の高精度なConvNet
	# ネットワーク構成は下記の通り
    #    conv - relu - conv- relu - pool -
    #    conv - relu - conv- relu - pool -
    #    conv - relu - conv- relu - pool -
    #    affine - relu - dropout - affine - dropout - softmax

    def initialize(input_dim:[1, 28, 28],
                 conv_param_1:{'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2:{'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_3:{'filter_num':32, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_4:{'filter_num':32, 'filter_size':3, 'pad':2, 'stride':1},
                 conv_param_5:{'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_6:{'filter_num':64, 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size:50, output_size:10)
        # 重みの初期化===========
        # 各層のニューロンひとつあたりが、前層のニューロンといくつのつながりがあるか（TODO:自動で計算する）
        pre_node_nums = Numo::DFloat[1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size]
        wight_init_scales = Numo::NMath.sqrt(2.0 / pre_node_nums)  # ReLUを使う場合に推奨される初期値
        
        @params = {}
        pre_channel_num = input_dim[0]
        [conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6].each_with_index{|conv_param, idx|
            @params['w' + (idx+1).to_s] = wight_init_scales[idx] * Numo::DFloat.new(1, conv_param[:filter_num],
            								 pre_channel_num, conv_param[:filter_size], conv_param[:filter_size]).rand_norm
            @params['b' + (idx+1).to_s] = Numo::DFloat.zeros(1, conv_param[:filter_num])
            pre_channel_num = conv_param[:filter_num]
        }
        @params['w7'] = wight_init_scales[6] * Numo::DFloat.new(1, 64*4*4, hidden_size).rand_norm
        @params['b7'] = Numo::DFloat.zeros(1, hidden_size)
        @params['w8'] = wight_init_scales[7] * Numo::DFloat.new(1, hidden_size, output_size).rand_norm
        @params['b8'] = Numo::DFloat.zeros(1, output_size)

        # レイヤの生成===========
        @layers = []
        @layers << Convolution.new(w:@params['w1'][0, false], b:@params['b1'][0, false], stride:conv_param_1[:stride], pad:conv_param_1[:pad])
        @layers << Relu.new
        @layers << Convolution.new(w:@params['w2'][0, false], b:@params['b2'][0, false], stride:conv_param_2[:stride], pad:conv_param_2[:pad])
        @layers << Relu.new
        @layers << Pooling.new(pool_h:2, pool_w:2, stride:2)
        @layers << Convolution.new(w:@params['w3'][0, false], b:@params['b3'][0, false], stride:conv_param_3[:stride], pad:conv_param_3[:pad])
        @layers << Relu.new
        @layers << Convolution.new(w:@params['w4'][0, false], b:@params['b4'][0, false], stride:conv_param_4[:stride], pad:conv_param_4[:pad])
        @layers << Relu.new
        @layers << Pooling.new(pool_h:2, pool_w:2, stride:2)
        @layers << Convolution.new(w:@params['w5'][0, false], b:@params['b5'][0, false], stride:conv_param_5[:stride], pad:conv_param_5[:pad])
        @layers << Relu.new
        @layers << Convolution.new(w:@params['w6'][0, false], b:@params['b6'][0, false], stride:conv_param_6[:stride], pad:conv_param_6[:pad])
        @layers << Relu.new
        @layers << Pooling.new(pool_h:2, pool_w:2, stride:2)
        @layers << Affine.new(w:@params['w7'][0, false], b:@params['b7'][0, false])
        @layers << Relu.new
        @layers << Dropout.new(dropout_ratio:0.5)
        @layers << Affine.new(w:@params['w8'][0, false], b:@params['b8'][0, false])
        @layers << Dropout.new(dropout_ratio:0.5)
        
        @last_layer = SoftmaxWithLoss.new
    end

    def predict(x:, train_flag:false)
 	   	@layers.inject(x) {|x, layer| x = layer.forward(x:x, train_flag:train_flag) }
    end

    def loss(x:, t:, train_flag:false)	#損失関数を求める
	    y = predict(x:x, train_flag:train_flag)
        return @last_layer.forward(x:y, t:t)
    end

    def accuracy(x:, t:, batch_size:100)
        t = t.max_index(1) % t.shape[1]  if t.ndim != 1
        
        acc = 0.0
        
        (x.shape[0] / batch_size).to_i.times{|i|
        	tx = x[i*batch_size...(i+1)*batch_size, false]
        	tt = t[i*batch_size...(i+1)*batch_size, false]
        	y = predict(x:tx, train_flag:false)
        	y = y.max_index(1) % y.shape[1]
            acc += y.eq(tt).cast_to(Numo::UInt16).sum
        }
        return acc / x.shape[0]
	end

    def gradient(x:, t:)	# 勾配を求める（誤差逆伝搬法）
        # forward
 	    loss(x:x, t:t, train_flag:true)

        # backward
        dout = 1
        dout = @last_layer.backward(dout: dout)

	    layers = @layers.reverse
    	layers.inject(dout) {|dout, layer|
	        dout = layer.backward(dout: dout)
    	}

        # 設定
        grads = {}
        [0, 2, 5, 7, 10, 12, 15, 18].each_with_index{|layer_idx, i|
            grads['w' + (i+1).to_s] = @layers[layer_idx].dw
            grads['b' + (i+1).to_s] = @layers[layer_idx].db
		}

        return grads
    end

    def save_params(file_name = "deep_convnet_params.dump")
  		puts "Creating dump file ..."
  		File.open(file_name, "w"){|f| Marshal.dump(@params, f)}
		puts "Done!"
	end

    def load_params(file_name = "deep_convnet_params.dump")
  		File.open("deep_convnet_params.dump", "r"){|f| @params = Marshal.load(f)}
        [0, 2, 5, 7, 10, 12, 15, 18].each_with_index{|layer_idx, i|
            @layers[layer_idx].w = @params['w' + (i+1).to_s][0, false]
            @layers[layer_idx].b = @params['b' + (i+1).to_s][0, false]
        }
	end
end