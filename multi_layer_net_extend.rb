require 'numo/narray'
require './numerical_gradient.rb'
require './layers_3.rb'

class MultiLayerNet
	attr_reader :use_batchnorm
  def initialize(input_size:, hidden_size_list:, output_size:, activation: :relu, weight_init_std: :relu,
  					 weight_decay_lambda: 0, use_dropout: false, dropout_ratio: 0.5, use_batchnorm: false)
  	  												# activation : 'relu' or 'sigmoid'
  	  												# weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
  													#       'relu'または'he'を指定した場合は「Heの初期値」を設定
													#       'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    @input_size          = input_size				# 入力サイズ（MNISTの場合は784）
    @output_size         = output_size				# 出力サイズ（MNISTの場合は10）
    @hidden_size_list    = hidden_size_list			# 隠れ層のニューロンの数のリスト（e.g. [100, 100, 100]）
    @hidden_layer_num    = hidden_size_list.size
    @weight_decay_lambda = weight_decay_lambda		# Weight Decay（L2ノルム）の強さ
    @use_dropout = use_dropout
    @use_batchnorm = use_batchnorm
    @params              = {}

    # 重みの初期化
    init_weight(weight_init_std)

    # レイヤの生成
    activation_layer = { sigmoid: Sigmoid, relu: Relu }
    @layers = {}
    (1..@hidden_layer_num).each do |idx|
      @layers["Affine#{idx}"] = Affine.new(w: @params["w#{idx}"][0, false], b: @params["b#{idx}"][0, false])
      if @use_batchnorm
      	  @params["gamma" + "#{idx}"] = Numo::DFloat.ones(1, hidden_size_list[idx - 1])
      	  @params["beta" + "#{idx}"] = Numo::DFloat.zeros(1, hidden_size_list[idx - 1])
      	  @layers['BatchNorm' + "#{idx}"] = BatchNormalization.new(gamma: @params["gamma" + "#{idx}"][0, false], beta: @params["beta" + "#{idx}"][0, false])
      end
      @layers["Activation_function#{idx}"] = activation_layer[activation].new
      @layers['Dropout' + "#{idx}"] = Dropout.new(dropout_ratio: dropout_ratio)  if @use_dropout
    end

    idx = @hidden_layer_num + 1
    @layers["Affine#{idx}"] = Affine.new(w: @params["w#{idx}"][0, false], b: @params["b#{idx}"][0, false])

    @last_layer = SoftmaxWithLoss.new
  end

  def params
    @params
  end

  def init_weight(weight_init_std)
    all_size_list = [@input_size] + @hidden_size_list + [@output_size]
    (1..(all_size_list.size - 1)).each do |idx|
      scale = weight_init_std
      if %i(relu he).include?(weight_init_std)
        scale = Numo::DFloat::Math.sqrt(2.0 / all_size_list[idx - 1])
      elsif %i(sigmoid xavier).include?(weight_init_std)
        scale = Numo::DFloat::Math.sqrt(1.0 / all_size_list[idx - 1])
      end

      Numo::NArray.srand
      @params["w#{idx}"] = scale * Numo::DFloat.new(1, all_size_list[idx - 1], all_size_list[idx]).rand_norm
      @params["b#{idx}"] = Numo::DFloat.zeros(1, all_size_list[idx])
    end
  end

  def predict(x:, train_flag: false)
    @layers.values.inject(x) do |x, layer|
      x = layer.forward(x: x, train_flag: train_flag)
    end
  end

  # x: 入力データ, t: 教師データ
  def loss(x:, t:, train_flag: false)
    y = predict(x: x, train_flag: train_flag)

    weight_decay = 0
    (1..(@hidden_layer_num + 1)).each do |idx|
      w = @params["w#{idx}"][0, false]
      weight_decay += 0.5 * @weight_decay_lambda * (w ** 2).sum
    end
    @last_layer.forward(x: y, t: t) + weight_decay
  end

  def accuracy(x:, t:)
    y = predict(x: x, train_flag: false)
    y = y.max_index(1) % y.shape[1]
    if t.ndim != 1
      t = t.max_index(1) % t.shape[1]
    end

    y.eq(t).cast_to(Numo::UInt16).sum / x.shape[0].to_f
  end

  def numerical_gradients(x:, t:)    # 勾配を求める（数値微分）
      loss_w = lambda { loss(x: x, t: t, train_flag: true) }
      grads = {}
      (1..@hidden_layer_num + 1).each{|idx|
        grads["w#{idx}"] = numerical_gradient(loss_w, @params["w#{idx}"][0, false])
        grads["b#{idx}"] = numerical_gradient(loss_w, @params["b#{idx}"][0, false])
        if @use_batchnorm and idx != @hidden_layer_num + 1
        	grads["gamma#{idx}"] = numerical_gradient(loss_w, @params["gamma#{idx}"])
        	grads["beta#{idx}"] = numerical_gradient(loss_w, @params["beta#{idx}"])
        end
	  }
    return grads
  end

  def gradient(x:, t:)
    # forward
    loss(x: x, t: t, train_flag: true)

    # backward
    dout = 1
    dout = @last_layer.backward(dout: dout)

    layers = @layers.values.reverse
    layers.inject(dout) {|dout, layer|
      dout = layer.backward(dout: dout)
    }

    grads = {}
    (1..(@hidden_layer_num + 1)).each do |idx|
      grads["w#{idx}"] = @layers["Affine#{idx}"].dw + @weight_decay_lambda * @layers["Affine#{idx}"].w
      grads["b#{idx}"] = @layers["Affine#{idx}"].db
      if @use_batchnorm and (idx != @hidden_layer_num + 1)
      	grads["gamma#{idx}"] = @layers["BatchNorm#{idx}"].dgamma
       	grads["beta#{idx}"] = @layers["BatchNorm#{idx}"].dbeta
      end
    end

    grads
  end
end

