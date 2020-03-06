import sys
import os


def localdir(pa):
    return os.path.join(os.path.dirname(__file__), str(pa))


def product(num, *nums):
    val = 1
    inputs = ([x for x in num] if hasattr(num, '__iter__') else [num])
    inputs.extend(nums)
    for numb in inputs: val *= numb
    return val


def get_conv_output_size(channel_vol, kernel, padding=0, stride=1):
    return ((channel_vol - kernel + 2*padding) // stride) + 1


def get_convT_output_size(channel_vol, kernel, padding=0, stride=1):
    return ((channel_vol - 1) * stride) + kernel - 2*padding


def flat2matrix(tens, batch_size=1, *args, **kwargs):
    mat_size = int(tens.shape[0] ** 0.5)
    return tens.reshape(batch_size, 1, mat_size, mat_size)


def matrix2flat(tens, *args, **kwargs):
    return tens.flatten()