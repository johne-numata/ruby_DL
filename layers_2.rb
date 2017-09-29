require './functions.rb'

class Relu
  def initialize
    @mask = nil
  end

  def forward(x:)
    @mask = (x <= 0)
    out = x.copy
    out[@mask] = 0
    out
  end

  def backward(dout:)
    dout[@mask] = 0
    dout
  end
end

class Sigmoid
  def initialize
    @out = nil
  end

  def forward(x:)
    @out = sigmoid(x)
    @out
  end

  def backword(dout:)
    dout * (1.0 - @out) * @out
  end
end

class Affine
  attr_reader :w, :dw, :db 
  def initialize(w:, b:)
    @w = w
    @b = b

    @x = nil
    @original_x_shape = nil

    # �d�݁E�o�C�A�X�p�����[�^�̔���
    @dw = nil
    @db = nil
  end

  def forward(x:)
    # �e���\���Ή�
    @original_x_shape = x.shape
    @x = x.reshape(x.shape[0], nil)
    @x.dot(@w) + @b
  end

  def backward(dout:)
    @dw = @x.transpose.dot(dout)
    @db = dout.sum(0)
    return  dout.dot(@w.transpose).reshape(*@original_x_shape)	# x�̔����l
  end
end

class SoftmaxWithLoss
  def initialize
    @loss = nil
    @y = nil # softmax�̏o��
    @t = nil # ���t�f�[�^
  end

  def forward(x:, t:)
    @t = t
    @y = softmax(x)
    @loss = cross_entropy_error(@y, @t)

    @loss
  end

  def backward(dout: 1)
    batch_size = @t.shape[0]
    if @t.size == @y.size # ���t�f�[�^��on-hot-vector�̏ꍇ
      return (@y - @t) / batch_size
    end

    dx = @y.copy

    (0..(batch_size - 1)).to_a.zip(@t).each do |index_array|
      dx[*index_array] -= 1
    end

    dx / batch_size
  end
end