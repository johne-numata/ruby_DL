require 'numo/narray'
require './numerical_gradient.rb'
require './layers_2.rb'

class TwoLayerNet
  attr_reader :params, :layers
  def initialize(input_size:, hidden_size:, output_size:, weight_init_std: 0.01)
    # 重みの初期化、Layerから間接参照するため無理矢理次元を増やす
    Numo::NArray.srand
    @params = {
      w1: weight_init_std * Numo::DFloat.new(1, input_size, hidden_size).rand_norm,
      b1: Numo::DFloat.zeros(1, hidden_size),
      w2: weight_init_std * Numo::DFloat.new(1, hidden_size, output_size).rand_norm,
      b2: Numo::DFloat.zeros(1, output_size)
    }

    # レイヤの生成
    @layers = {
      affine1: Affine.new(w: @params[:w1][0, false], b: @params[:b1][0, false]),
      relu1:   Relu.new,
      affine2: Affine.new(w: @params[:w2][0, false], b: @params[:b2][0, false])
    }
    @last_layer = SoftmaxWithLoss.new
  end

  def predict(x:)
    @layers.values.inject(x) do |x, layer|
      x = layer.forward(x: x)
    end
  end

  # x: 入力データ, t: 教師データ
  def loss(x:, t:)
    y = predict(x: x)
    @last_layer.forward(x: y, t: t)
  end

  def accuracy(x:, t:)
    y = predict(x: x)
    y = y.max_index(1) % 10
    if t.ndim != 1
      t = t.max_index(1) % 10
    end

    y.eq(t).cast_to(Numo::UInt16).sum / x.shape[0].to_f
  end

  def numerical_gradients(x:, t:)
    loss_w = lambda { loss(x: x, t: t) }

    {
      w1: numerical_gradient(loss_w, @params[:w1][0, false] ),
      b1: numerical_gradient(loss_w, @params[:b1][0, false] ),
      w2: numerical_gradient(loss_w, @params[:w2][0, false] ),
      b2: numerical_gradient(loss_w, @params[:b2][0, false] )
    }
  end

  def gradient(x:, t:)
    # forward
    loss(x: x, t: t)

    # backward
    dout = 1
    dout = @last_layer.backward(dout: dout)

    layers = @layers.values.reverse
    layers.inject(dout) do |dout, layer|
      dout = layer.backward(dout: dout)
    end

    {
      w1: @layers[:affine1].dw,
      b1: @layers[:affine1].db,
      w2: @layers[:affine2].dw,
      b2: @layers[:affine2].db
    }
  end
end

