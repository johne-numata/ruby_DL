require 'numo/narray'

=begin
def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]
=end


def conv_output_size(input_size:, filter_size:, stride: 1, pad: 0)
    return (input_size + 2 * pad - filter_size) / stride + 1
end


def im2col(input_data:, filter_h:, filter_w:, stride: 1, pad: 0)
#   Parameters
#   ----------
#   input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
#   filter_h : フィルターの高さ
#   filter_w : フィルターの幅
#   stride : ストライド
#   pad : パディング

#   Returns
#   -------
#   col : 2次元配列

    n, c, h, w = input_data.shape
    out_h = ((h + 2 * pad - filter_h) / stride).to_i + 1
    out_w = ((w + 2 * pad - filter_w) / stride).to_i + 1

#    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant') # Pythonの場合
	img = Numo::DFloat.zeros(n, c, h + 2 * pad, w + 2 * pad)
	img[0...n, 0...c, pad...(h + pad), pad...(h + pad)] = input_data
    col = Numo::DFloat.zeros(n, c, filter_h, filter_w, out_h, out_w)

    filter_h.times{|y|
        y_max = y + stride * out_h - 1
        filter_w.times{|x|
            x_max = x + stride * out_w - 1
            col[0..-1, 0..-1, y, x, 0..-1, 0..-1] = img[0..-1, 0..-1, y.step(y_max, stride).to_a, x.step(x_max, stride).to_a]
        }
	}
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n * out_h * out_w, true)
    return col
end

def col2im(col:, input_shape:, filter_h:, filter_w:, stride: 1, pad: 0)
#    Parameters
#    ----------
#    col :
#    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
#    filter_h :
#    filter_w
#    stride
#    pad

    n, c, h, w = input_shape
    out_h = ((h + 2 * pad - filter_h) / stride).to_i + 1
    out_w = ((w + 2 * pad - filter_w) / stride).to_i + 1
    col = col.reshape(n, out_h, out_w, c, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = Numo::DFloat.zeros(n, c, h+2*pad+stride-1, w+2*pad+stride-1)
    filter_h.times{|y|
        y_max = y + stride*out_h-1
        filter_w.times{|x|
            x_max = x + stride*out_w-1
            img[0..-1, 0..-1, y.step(y_max, stride).to_a, x.step(x_max, stride).to_a] += col[0..-1, 0..-1, y, x, 0..-1, 0..-1]
        }
	}
    return img[0..-1, 0..-1, pad...(h + pad), pad...(w + pad)]
end
