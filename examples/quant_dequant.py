
def affine_quantize(r, scale, zero_point=0,bit_size=8):
    lower_bound = 0
    upper_bound = (2**bit_size) - 1
    r_quant = round(r/scale) + zero_point
    r_int = max(lower_bound, min(r_quant, upper_bound))
    return r_int

def symmetric_quantize(r, scale, zero_point=0,bit_size=8):
    upper_bound = (2**(bit_size - 1)) - 1
    lower_bound = -(2**(bit_size - 1))
    r_quant = round(r/scale) + zero_point
    r_int = max(lower_bound, min(r_quant, upper_bound))
    return r_int

def dequantize(q, scale, zero_point=0):
    return scale * (q - zero_point)


min_real_value, max_real_value = -1, 1
bit_size = 8
scale = (max_real_value - min_real_value) / ((2**bit_size) - 1)
zero_point = 0  # symmetric quantization (weight)
real_values = [-1, 0, 1]    # must be between 0 and 1


for real_value in real_values:
    assert (real_value >= min_real_value) and (real_value <= max_real_value) 
    quantized_value = symmetric_quantize(r=real_value, scale=scale, zero_point=zero_point, bit_size=bit_size)
    recovered_real_value = dequantize(q=quantized_value, scale=scale, zero_point=zero_point)
    print(f"real value: {real_value} \t symmetric quant: {quantized_value} \t recovered real value: {recovered_real_value} \t rounding error: {abs(real_value - recovered_real_value)}")

print()

min_real_value, max_real_value = 0, 1
bit_size = 8
scale = (max_real_value - min_real_value) / ((2**bit_size) - 1)
zero_point = 0  # affine quantization (weight)
real_values = [0, 0.5, 1]    # must be between 0 and 1


for real_value in real_values:
    assert (real_value >= min_real_value) and (real_value <= max_real_value) 
    quantized_value = affine_quantize(r=real_value, scale=scale, zero_point=zero_point, bit_size=bit_size)
    recovered_real_value = dequantize(q=quantized_value, scale=scale, zero_point=zero_point)
    print(f"real value: {real_value} \t affine quant: {quantized_value} \t recovered real value: {recovered_real_value} \t rounding error: {abs(real_value - recovered_real_value)}")