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

class Convolution
    def initialize(w:, b:, stride: 1, pad: 0)
        @w = w
        @b = b
        @stride = stride
        @pad = pad
        
        # 中間データ（backward時に使用）
        @x = None   
        @col = None
        @col_w = None
        
        # 重み・バイアスパラメータの勾配
        @dw = nil
        @db = nil
	end

    def forward(x:)
        fn, c, fh, fw = @w.shape
        n, c, h, w = x.shape
        out_h = 1 + ((h + 2*@pad - fh) / @stride).to_i
        out_w = 1 + ((w + 2*@pad - fw) / @stride).to_i

        col = im2col(x: x, filter_h: fh, filter_w: fw, stride: @stride, pad: @pad)
        col_w = @w.reshape(fn, true).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out
	end

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
end

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

