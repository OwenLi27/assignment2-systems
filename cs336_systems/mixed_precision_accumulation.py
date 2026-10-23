import torch

"""
四个结果体现了两种不同的误差来源：数值被存入低精度格式时的量化误差
以及低精度累加器每一步产生的舍入误差。

  1. FP32 输入 + FP32 累加器 → 10.0001
     最精确
     0.01 在二进制浮点数中不能被精确表示，但 FP32 有约 24 位
     有效二进制精度，所以累计 1000 次后的误差很小，约为
     +0.0001。

  2. FP16 输入 + FP16 累加器 → 9.9531 
     体现了量化误差和舍入误差
     这是误差最大的一组。问题不仅是 FP16 中的 0.01 不精确，更
     重要的是每次加法结果都必须重新舍入为 FP16。数值达到 8 附
     近时，相邻 FP16 数的间隔已经是 0.0078125,与要加的 0.01
     非常接近，因此每一步都会丢失相当一部分增量，误差不断累
     积，最终比 10 少约 0.0469。

  3. FP16 输入 + FP32 累加器 → 10.0021
     体现了量化误差，但避免了舍入误差
     FP16 中实际保存的 0.01 是:0.01000213623046875
     把它隐式提升到 FP32 不会恢复已经丢失的精度，但后续累加可以在
     FP32 中进行，避免每一步再次发生严重舍入。于是结果接近:
     1000 × 0.01000213623046875 = 10.002136...

  4. 显式转成 FP32 → 同样是 10.0021
     同3, 第四段显式转换没有改变结果。两者都保留了最初的
     FP16 量化误差，但避免了 FP16 累加误差。

  核心结论是：高精度累加器无法修复输入已经发生的量化误差，但
  能防止误差在每次求和时继续放大。因此混合精度训练通常允许矩
  阵乘法使用 FP16/BF16, 同时让求和、均值、方差等 reduction/
  accumulation 在 FP32 中完成。
"""

if __name__ == "__main__":
    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float32)
    print(s)

    s = torch.tensor(0,dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01,dtype=torch.float16)
    print(s)

    s = torch.tensor(0,dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01,dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)