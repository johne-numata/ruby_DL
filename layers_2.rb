require './functions.rb'

class Relu
  def initialize
    @mask = nil
  end

  def forward(x:, train_flag: nil)
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

  def forward(x:, train_flag: nil)
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

    # 重み・バイアスパラメータの微分
    @dw = nil
    @db = nil
  end

  def forward(x:, train_flag: nil)
    # テンソル対応
    @original_x_shape = x.shape
    @x = x.reshape(x.shape[0], true)
    @x.dot(@w) + @b
  end

  def backward(dout:)
    @dw = @x.transpose.dot(dout)
    @db = dout.sum(0)
    return  dout.dot(@w.transpose).reshape(*@original_x_shape)	# xの微分値
  end
end

class SoftmaxWithLoss
  def initialize
    @loss = nil
    @y = nil # softmaxの出力
    @t = nil # 教師データ
  end

  def forward(x:, t:, train_flag: nil)
    @t = t
    @y = softmax(x)
    @loss = cross_entropy_error(@y, @t)

    @loss
  end

  def backward(dout: 1)
    batch_size = @t.shape[0]
    if @t.size == @y.size # 教師データがon-hot-vectorの場合
      return (@y - @t) / batch_size
    end

    dx = @y.copy

    (0..(batch_size - 1)).to_a.zip(@t).each do |index_array|
      dx[*index_array] -= 1
    end

    dx / batch_size
  end
end

class Dropout
    def initialize(dropout_ratio: 0.5)
        @dropout_ratio = dropout_ratio
        @mask = nil
	end

    def forward(x:, train_flag: true)
        if train_flag
            @mask = Numo::DFloat.new(x.shape).rand > @dropout_ratio
            return x * @mask
        else
            return x * (1.0 - @dropout_ratio)
        end
	end

    def backward(dout:)
        return dout * (@mask ? @mask : 1.0)
	end
end

class BatchNormalization
	attr_reader :dgamma, :dbeta
    def initialize(gamma:, beta:, momentum: 0.9, running_mean: nil, running_var: nil)
        @gamma = gamma
        @beta = beta
        @momentum = momentum
        @input_shape = nil		 # Conv層の場合は4次元、全結合層の場合は2次元  

        # テスト時に使用する平均と分散
        @running_mean = running_mean
        @running_var = running_var  
        
        # backward時に使用する中間データ
        @batch_size = nil
        @xc = nil
        @std = nil
        @dgamma = nil
        @dbeta = nil
	end

    def forward(x:, train_flag: true)
        @input_shape = x.shape
        if x.ndim != 2
            n, c, h, w = x.shape
            x = x.reshape(n, true)
		end
        out = _forward(x: x, train_flag: train_flag)
        
        return out.reshape(*@input_shape)
	end

    def _forward(x:, train_flag:)
        if @running_mean.nil?
            n, d = x.shape
            @running_mean = Numo::DFloat.zeros(d)
            @running_var = Numo::DFloat.zeros(d)
        end
        if train_flag
            mu = x.mean(0)
            xc = x - mu
            var = (xc**2).mean(0)
            std = Numo::NMath.sqrt(var + 10e-7)
            xn = xc / std
            
            @batch_size = x.shape[0]
            @xc = xc
            @xn = xn
            @std = std
            @running_mean = @momentum * @running_mean + (1 - @momentum) * mu
            @running_var = @momentum * @running_var + (1 - @momentum) * var            
        else
            xc = x - @running_mean
            xn = xc / ((Numo::NMath.sqrt(@running_var + 10e-7)))
        end
        out = @gamma * xn + @beta 
        return out
    end

    def backward(dout:)
        if dout.ndim != 2
            n, c, h, w = dout.shape
            dout = dout.reshape(n, true)
		end
        dx = _backward(dout)

        dx = dx.reshape(*@input_shape)
        return dx
    end

    def _backward(dout)
        dbeta = dout.sum(0)
        dgamma = (@xn * dout).sum(0)
        dxn = @gamma * dout
        dxc = dxn / @std
        dstd = -((dxn * @xc) / (@std * @std)).sum(0)
        dvar = 0.5 * dstd / @std
        dxc += (2.0 / @batch_size) * @xc * dvar
        dmu = dxc.sum(0)
        dx = dxc - dmu / @batch_size
        
        @dgamma = dgamma
        @dbeta = dbeta
        
        return dx
	end
end
