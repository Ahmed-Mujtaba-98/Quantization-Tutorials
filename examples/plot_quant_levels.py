import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from scipy import stats

def calculate_sparsity(array):
    """
    Calculate the sparsity of an array.
    
    Parameters:
    array (numpy.ndarray): Input array to calculate the sparsity for.

    Returns:
    float: Sparsity of the array, which is the ratio of zero elements to the total elements.
    """
    # Convert the input to a numpy array if it isn't already one
    array = np.array(array)

    # Total number of elements in the array
    total_elements = array.size

    # Number of zero elements in the array
    zero_elements = np.count_nonzero(array == 0)

    # Sparsity is the ratio of zero elements to total elements
    sparsity = zero_elements / total_elements

    return f"{(sparsity*100):.2f}%"

def linear_quantize(x, n_bits=4):
    n_levels = 2**n_bits
    max, min = x.max(), x.min()
    delta = (max - min) / (2 ** n_bits - 1)
    zero_point = (- min / delta).round()
    # we assume weight quantization is always signed
    x_int = np.round(x / delta)
    x_quant = np.clip(x_int + zero_point, 0, n_levels - 1)
    x_float_q = (x_quant - zero_point) * delta
    return x_float_q, x_quant

def augq_quantize_int(data_tensor, n_bits):

    data_tensor = torch.tensor(data_tensor)
    n_levels = 2**n_bits
    max, min = data_tensor.max(), data_tensor.min()

    step = (max - min) / 4
    step_a = min + step
    step_b = step_a + step
    step_c = step_b + step

    # scale_a == scale_b == scale_c == scale_d      (All same i.e., uniform)
    scale_a = (step_a - min) / (n_levels - 1)
    scale_b = (step_b - step_a) / (n_levels - 1)
    scale_c = (step_c - step_b) / (n_levels - 1)
    scale_d = (max - step_c) / (n_levels - 1)

    zero_point_a = (-min / scale_a).round()
    zero_point_b = (-step_a / scale_b).round()
    zero_point_c = (-step_b / scale_c).round()
    zero_point_d = (-step_c / scale_d).round()
    # assert scale_a == scale_b == scale_c == scale_d

    scale = (scale_a + scale_b + scale_c + scale_d) / 4

    # print(step_a - min, " ", step_b - step_a, " ", step_c - step_b, " ", max - step_c)
    # print(data_tensor)

    data_quant_a = torch.where((min <= data_tensor) & (data_tensor < step_a),
                              data_tensor.div(scale).round().add(zero_point_a).clamp(0, n_levels - 1),
                              0
                              )
    data_quant_b = torch.where((step_a < data_tensor) & (data_tensor < step_b),
                              data_tensor.div(scale).round().add(zero_point_b).clamp(0, n_levels - 1),
                              0
                              )
    data_quant_c = torch.where((step_b < data_tensor) & (data_tensor < step_c),
                              data_tensor.div(scale).round().add(zero_point_c).clamp(0, n_levels - 1),
                              0
                              )
    data_quant_d = torch.where((step_c < data_tensor) & (data_tensor <= max),
                              data_tensor.div(scale).round().add(zero_point_d).clamp(0, n_levels - 1),
                              0
                              )

    # print(data_quant_a)
    # print(data_quant_b)
    # print(data_quant_c)
    # print(data_quant_d)

    data_quant = data_quant_a + data_quant_b + data_quant_c + data_quant_d
    
    # print(data_quant)
    # print(calculate_sparsity(data_quant_a), calculate_sparsity(data_quant_b), calculate_sparsity(data_quant_c), calculate_sparsity(data_quant_d))

    return data_quant, data_quant_a, data_quant_b, data_quant_c, data_quant_d

def divide_range(data_tensor):

    data_tensor = torch.tensor(data_tensor)
    max, min = data_tensor.max(), data_tensor.min()

    step = (max - min) / 4
    step_a = min + step
    step_b = step_a + step
    step_c = step_b + step

    data_real_a = torch.where((min <= data_tensor) & (data_tensor < step_a),
                              data_tensor,
                              0
                              )
    data_real_b = torch.where((step_a < data_tensor) & (data_tensor < step_b),
                              data_tensor,
                              0
                              )
    data_real_c = torch.where((step_b < data_tensor) & (data_tensor < step_c),
                              data_tensor,
                              0
                              )
    data_real_d = torch.where((step_c < data_tensor) & (data_tensor <= max),
                              data_tensor,
                              0
                              )
    data_real = data_real_a + data_real_b + data_real_c + data_real_d
    
    # print(data_real == data_tensor)

    return data_real_a, data_real_b, data_real_c, data_real_d



bin_size = 400
bit_size = 4

# weight = np.random.normal(size=(16,4,5,5))
weight = np.random.normal(size=(1,1,3,3))
kde = stats.gaussian_kde(weight.reshape(-1))
xx = np.linspace(weight.max()+1, weight.min()-1, 5*5*4*3)


data_real_a, data_real_b, data_real_c, data_real_d = divide_range(weight.reshape(-1))
_ , data_quant_a = linear_quantize(data_real_a, n_bits=bit_size)
_ , data_quant_b = linear_quantize(data_real_b, n_bits=bit_size)
_ , data_quant_c = linear_quantize(data_real_c, n_bits=bit_size)
_ , data_quant_d = linear_quantize(data_real_d, n_bits=bit_size)

# print(len(data_quant_a) == len(data_quant_b) == len(data_quant_c) == len(data_quant_d))

augq_weights_four, augq_data_quant_a, augq_data_quant_b, augq_data_quant_c, augq_data_quant_d = augq_quantize_int(weight.reshape(-1), n_bits=bit_size)

print(data_quant_a)
print(augq_data_quant_a)

# print(data_quant_a == augq_data_quant_a)

# # Create separate figures for each plot
# fig1, ax1 = plt.subplots(figsize=(8, 6))
# fig2, ax2 = plt.subplots(figsize=(8, 6))
# fig3, ax3 = plt.subplots(figsize=(8, 6))
# fig4, ax4 = plt.subplots(figsize=(8, 6))

# # # Plot the first plot
# ax1.plot(sorted(data_real_a), sorted(data_quant_a))
# ax1.set_xlabel('data_real_a')
# ax1.set_ylabel('Quantized Weight')
# ax1.set_title(f'AUGQ data_real_a')
# ax1.legend()

# # Plot the second plot
# ax2.plot(sorted(data_real_b), sorted(data_quant_b))
# ax2.set_xlabel('data_real_b')
# ax2.set_ylabel('Quantized Weight')
# ax2.set_title(f'AUGQ data_real_b')
# ax2.legend()

# # Plot the third plot
# ax3.plot(sorted(data_real_c), sorted(data_quant_c))
# ax3.set_xlabel('data_real_c')
# ax3.set_ylabel('Quantized Weight')
# ax3.set_title(f'AUGQ data_real_c')
# ax3.legend()

# # Plot the fourth plot
# ax4.plot(sorted(data_real_d), sorted(data_quant_d))
# ax4.set_xlabel('data_real_d')
# ax4.set_ylabel('Quantized Weight')
# ax4.set_title(f'AUGQ data_real_d')
# ax4.legend()

# plt.show()













# channels = weight.shape[0]
# channel_quant_four = [0] * channels
# channel_int_four = [0] * channels

# for i in range(channels):
#   channel_quant_four[i], channel_int_four[i] = linear_quantize(weight[i].reshape(-1), n_bits=bit_size)


# weight = weight.reshape(-1)
# linear_weights_four, int_four = linear_quantize(weight, n_bits=bit_size)
# augq_weights_four, _, _, _, _ = augq_quantize_int(weight, n_bits=bit_size)









# plt.figure(figsize=(8, 6))  # Adjust figure size if needed
# ax = plt.gca()  # Get current axes

# ax.plot(sorted(weight), sorted(int_four), label=f'layer-wise ({bit_size}-bit)')
# ax.plot(sorted(weight), sorted(np.array(channel_int_four).reshape(-1)), label=f'Channel-wise ({bit_size}-bit)')
# ax.plot(sorted(weight), sorted(augq_weights_four), label=f'AUGQ ({bit_size}-bit)')
# ax.legend()
# plt.savefig('uniform_n3.png', format='png', dpi=1000, bbox_inches='tight')
# plt.show()


# # Create separate figures for each plot
# fig1, ax1 = plt.subplots(figsize=(8, 6))
# fig2, ax2 = plt.subplots(figsize=(8, 6))
# fig3, ax3 = plt.subplots(figsize=(8, 6))

# # Plot the first plot
# ax1.plot(sorted(weight), sorted(int_four))
# ax1.set_xlabel('Weight')
# ax1.set_ylabel('Quantized Weight')
# ax1.set_title(f'Layer-wise Quantization ({bit_size}-bit)')
# ax1.legend()

# # Plot the second plot
# ax2.plot(sorted(weight), sorted(np.array(channel_int_four).reshape(-1)))
# ax2.set_xlabel('Weight')
# ax2.set_ylabel('Quantized Weight')
# ax2.set_title(f'Channel-wise Quantization ({bit_size}-bit)')
# ax2.legend()

# # Plot the third plot
# ax3.plot(sorted(weight), sorted(augq_weights_four))
# ax3.set_xlabel('Weight')
# ax3.set_ylabel('Quantized Weight')
# ax3.set_title(f'AUGQ Quantization ({bit_size}-bit)')
# ax3.legend()
# plt.show()




























# font_size = 15
# fig, axs = plt.subplots(1, 3, figsize=(20, 5))
# # fig.suptitle('Fixed Quantization')

# axs[0].plot(xx, kde(xx))
# axs[0].set_title("Real Weight Distribution", fontsize=font_size)
# axs[0].set_xlabel('real values', fontsize=font_size)
# axs[0].set_ylabel('count', fontsize=font_size)

# axs[1].hist(linear_weights_two, bins=bin_size, color = 'r')
# axs[1].set_title("PL Granularity @ (2-bit)", fontsize=font_size)
# axs[1].set_xlabel('quantized values', fontsize=font_size)
# axs[1].set_ylabel('count', fontsize=font_size)

# axs[2].hist(linear_weights_four, bins=bin_size, color = 'r')
# axs[2].set_title("PL Granularity @ (4-bit)", fontsize=font_size)
# axs[2].set_xlabel('quantized values', fontsize=font_size)
# axs[2].set_ylabel('count', fontsize=font_size)

# plt.savefig('uniform_n3.png', format='png', dpi=1000, bbox_inches='tight')
# plt.show()