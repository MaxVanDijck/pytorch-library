import torch
import torch.nn as nn

def _make_divisible(value, divisor, minValue=None):
    if minValue is None:
        minValue = divisor
    newValue = max(minValue, int(value + divisor/2) // divisor * divisor)
    if newValue < 0.9 * value:
        newValue += divisor
    return newValue