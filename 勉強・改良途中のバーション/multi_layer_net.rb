require 'numo/narray'
require './numerical_gradient.rb'
require './layers_2.rb'

class MultiLayerNet
  def initialize(input_size:, hidden_size_list:, output_size:, activation: :relu, weight_init_std: :relu, weight_decay_lambda: 0)
  	  												# activation : 'relu' or 'sigmoid'
  	  												# weight_init_std : �d�݂̕W���΍����w��ie.g. 0.01�j
  													#       'relu'�܂���'he'���w�肵���ꍇ�́uHe�̏����l�v��ݒ�
													#       'sigmoid'�܂���'xavier'���w�肵���ꍇ�́uXavier�̏����l�v��ݒ�
    @input_size          = input_size				# ���̓T�C�Y�iMNIST�̏ꍇ��784�j
    @output_size         = output_size				# �o�̓T�C�Y�iMNIST�̏ꍇ��10�j
    @hidden_size_list    = hidden_size_list			# �B��w�̃j���[�����̐��̃��X�g�ie.g. [100, 100, 100]�j
    @hidden_layer_num    = hidden_size_list.size
    @weight_decay_lambda = weight_decay_lambda		# Weight Decay�iL2�m�����j�̋���
    @params              = {}

    # �d�݂̏�����
    init_weight(weight_init_std)

    # ���C���̐���
    activation_layer = {
      sigmoid: Sigmoid,
      relu:    Relu
    }
    @layers = {}
    (1..@hidden_layer_num).each do |idx|
      @layers["Affine#{idx}"] = Affine.new(w: @params["w#{idx}"][0, false], b: @params["b#{idx}"][0, false])
      @layers["Activation_function#{idx}"] = activation_layer[activation].new
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

  def predict(x:)
    @layers.values.inject(x) do |x, layer|
      x = layer.forward(x: x)
    end
  end

  # x: ���̓f�[�^, t: ���t�f�[�^
  def loss(x:, t:)
    y = predict(x: x)

    weight_decay = 0
    (1..(@hidden_layer_num + 1)).each do |idx|
      w = @params["w#{idx}"][0, false]
      weight_decay += 0.5 * @weight_decay_lambda * (w ** 2).sum
    end
    @last_layer.forward(x: y, t: t) + weight_decay
  end

  def accuracy(x:, t:)
    y = predict(x: x)
    y = y.max_index(1)
    if t.ndim != 1
      t = t.max_index(1) % t.shape[1]
      y = y % t.shape[1]
  	else
      t = t.max_index(1) % t.size
      y = y % t.size
    end

    y.eq(t).cast_to(Numo::UInt16).sum / x.shape[0].to_f
  end

  def numerical_gradients(x:, t:)    # ���z�����߂�i���l�����j
    loss_w = lambda { loss(x: x, t: t) }
      grads = {}
      (1..@hidden_layer_num + 1).each{|idx|
        grads["w#{idx}"] = numerical_gradient(loss_w, @params["w#{idx}"][0, false])
        grads["b#{idx}"] = numerical_gradient(loss_w, @params["b#{idx}"][0, false])
	  }
    return grads
  end

  def gradient(x:, t:)
    # forward
    loss(x: x, t: t)

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
    end

    grads
  end
end

