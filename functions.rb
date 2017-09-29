#! ruby -Ks
require 'numo/narray'
require './plot_lib'

def step(x)
	return x > 0
end


def sigmoid(x)
    return 1 / (1 + Numo::DFloat::Math.exp(-x))    
end


def sigmoid_grad(x)
    return (1.0 - sigmoid(x)) * sigmoid(x)
end


def relu(x)
	y = x.copy
	y[y<0] = 0
    return y
end


def relu_grad(x)
    grad = Numo::DFloat.zeros(x.shape)
    grad[x >= 0] = 1
    return grad
end


def softmax(a)
    if a.ndim == 2
        a = a.transpose
        a = a - a.max(0)
        y =  Numo::DFloat::Math.exp(a) / Numo::DFloat::Math.exp(a).sum(0)
        return y.transpose 
	end
	max = a.max
	exp_a = Numo::NMath.exp(a - max)
	sum_exp_a = exp_a.sum
	return exp_a / sum_exp_a
end


def mean_squared_error(y, t)
    return  0.5 * ((y - t)**2).sum
end


def cross_entropy_error(y, t)
  if y.ndim == 1
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  end

  # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
  if t.size == y.size
    t = t.max_index(1) % t.shape[1]
  end

  batch_size = y.shape[0]
  target_data = (0..(batch_size - 1)).to_a.zip(t).map do |index_array|
    y[*index_array]
  end
  -Numo::DFloat::Math.log(target_data).sum / batch_size
end


def softmax_loss(x, t)
    y = softmax(x)
    return cross_entropy_error(y, t)
end


if $0 == __FILE__ then
#	x = Numo::DFloat.linspace(-5.0, 5.0, 101)		# 0が正確に０にならない
#	x = Numo::DFloat.new(101).seq(-5.0, 0.1)		# 0が正確に０にならない
	x = Numo::DFloat.new(101).seq(-50) /10
	y1 = step(x)
	y2 = sigmoid(x)
	y3 = relu(x)
	make_plot_2d([[x, y1, {w:'lines', t: "step"}],
				 [x, y2, {w:'lines', t: "sigmoid"}],
				 [x, y3, {w:'lines', t: "relu"}],
				 ],title:"", yrange:"[-0.1:5.1]")
	print "#{x[50]}, #{y1[50]}, #{x[51]}, #{y1[51]}\n\n"

	temp = Numo::DFloat[1010, 1000, 990]
	p softmax(temp)
end
