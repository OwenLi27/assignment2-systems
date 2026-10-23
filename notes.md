## benchmarking_mixed_precision
(a) data type
```
class ToyModel(nn.Module):
def __init__(self, in_features: int, out_features: int):
super().__init__()
self.fc1 = nn.Linear(in_features, 10, bias=False)
self.ln = nn.LayerNorm(10)
self.fc2 = nn.Linear(10, out_features, bias=False)
self.relu = nn.ReLU()
def forward(self, x):
x = self.relu(self.fc1(x))
x = self.ln(x)
x = self.fc2(x)
return x
```
三个原则:
- autocast 通常不会永久改变模型参数本身的 dtype。
- autocast 是按“算子”决定计算和输出 dtype，不是把上下文内所有tensor一律变成 FP16。
- 参数梯度通常需要与其对应参数保持兼容的 dtype。

* the model parameters within the autocast context? - fp32, 模型内部的参数不随意发生改变
* the output of the first feed-forward layer (ToyModel.fc1)? - fp16, linear矩阵乘法对精度不敏感，输出可以转换为fp16.linear属于适合autocase的算子
* the output of layer norm (ToyModel.ln)? - fp32, norm包含reduction操作, 属于数值敏感操作, 需要保留fp32
* the model’s predicted logits? - fp16, fc2仍然是linear矩阵乘法, 可以使用autocast
* the loss? - fp32, 计算cross entropy包含reduction操作, 属于数值敏感操作，需要保留fp32
* the model’s gradients? - fp32, fp32的模型参数最终对应的.grad也是fp32

(b) 
$$
  \mu=\operatorname{mean}(x),\qquad
$$ 

$$ \sigma^2=\operatorname{mean}((x-\mu)^2),\qquad $$

$$  y=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma+\beta
$$ 
其中最敏感的是：
  - 计算均值和方差时需要对大量元素进行
    reduction，低精度舍入误差会累积。
  - x - mean(x) 可能是两个接近数字相减，容易发
    生有效数字损失。
  - 平方在 FP16 中可能溢出，较小的方差或
    epsilon 也可能下溢。
  - 除法会进一步放大方差估计中的误差。

比较 FP16 和 BF16：

类型 | 指数位  |  尾数位  |  主要特点
--- | ----- | -----|----- |
FP16 | 5   | 10  | 精度稍高，但动态范围小
BF16 |  8  | 7    | 与 FP32 动态范围相近，但有效精度更低
FP32 | 8  |   23   |  同时具有较大范围和较高精度

因此，BF16 大幅缓解了 FP16 的溢出和下溢问题，但没有解决 reduction、相消和有限尾数精度的问题。实践中的稳妥结论是：**即使用 BF16，LayerNorm 的均值、方差等统计量仍适合用 FP32计算或累加，然后再按需要把输出转换回低精度。** 区别在于 FP16 同时面临动态范围和精度问题，而 BF16 主要面临精度问题。