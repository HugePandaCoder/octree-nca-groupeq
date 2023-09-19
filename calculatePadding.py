#o = output
p = 1#padding
k = 3#kernel_size
s = 1#stride
d = 3#dilation
o = [i + 2*p - k - (k-1)*(d-1)]/s + 1