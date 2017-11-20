#! ruby -Ks
require 'tk'
require './mnist.rb'
#require tkextlib/tkimg/jpeg"
require 'tk/image'

x_train, t_train, x_test, t_test = load_mnist

img = x_train[0, true]
label = t_train[0]
puts img.shape
puts img.max
puts img.min
puts label


#x_train, t_train, x_test, t_test = load_mnist(normalize: false, flatten: false, one_hot_label: false)
#p x_train.shape
#p x_train[0, true, true]
width = 28
height = 28
img = (img.reshape(28,28) * 255.0).cast_to(Numo::UInt8)
header = "P5\n#{width} #{height}\n255\n"
#ppm_image = (header + x_train[0, true, true].to_string).encode!("ascii-8bit")
ppm_image = (header + img[true, true].to_string).encode!("ascii-8bit")
tkimg = TkPhotoImage.new(data: ppm_image)
TkLabel.new(image: tkimg).pack
Tk.mainloop
