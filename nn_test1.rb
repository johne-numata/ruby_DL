require 'numo/narray'

def sigmoid(x)
	return 1 / (1 + Numo::NMath.exp(-x))
end

def relu(x)
	return Numo::NMath.max(0, x)
end

def identity_function(x)
	return x
end

def softmax(x)
	c = Numo::NMath(x)
	exp_a = Numo::NMath.exp(a - c)
	sum_exp_a = Numo::NMath.sum(exp_a)
	return y = exp_a / sum_exp_a
end

def init_network
	network = {}
	network["W1"] = Numo::NArray[[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
	network["b1"] = Numo::NArray[0.1, 0.2, 0.3]
	network["W2"] = Numo::NArray[[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]
	network["b2"] = Numo::NArray[0.1, 0.2]
	network["W3"] = Numo::NArray[[0.1, 0.3], [0.2, 0.4]]
	network["b3"] = Numo::NArray[0.1, 0.2]
	return network
end

def forward(network, x)
	w1, w2, w3 = network["W1"], network["W2"], network["W3"]
	b1, b2, b3 = network["b1"], network["b2"], network["b3"]

	a1 = x.dot(w1) + b1
	z1 = sigmoid(a1)
	a2 = z1.dot(w2) + b2
	z2 = sigmoid(a2)
	a3 = z2.dot(w3) + b3
	return y = identity_function(a3)
end

def mean_squared_error(y, t)
	return 0.5 * Numo::NMath.sum((y - t)**2)
end

def cross_entropy_error(y, t)
	if y.ndim == 1
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)
	end
	bach_size = y.shape[0]
	
	delta = 1e-7
	return - Numo::NMath.sum(t * Numo::NMath.log(y + delta)) / bach_size # 教師データがone-hot表現の場合
end


network = init_network
x = Numo::NArray[1.0, 0.5]
y = forward(network, x)
p y
