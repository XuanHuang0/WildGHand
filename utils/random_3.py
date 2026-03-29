import random

# 原始三个数字
nums = [10, 0.3, 1.0]  # 替换为你自己的数字

# 扰动尺度
scales = [0.1, 0.3, 0.5]

# 生成三组结果
results = []
for scale in scales:
    perturbed = [x + random.uniform(-scale, scale)*x for x in nums]
    results.append(perturbed)

# 打印三组结果
for i, r in enumerate(results):
    print(f"扰动±{scales[i]}后的结果: {r}")
