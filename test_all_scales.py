# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

import paddle

# 多尺度测试
configs = [
    # (batch_size, hidden_size, dtype, name)
    (96, 64, "float16", "tiny_tiny"),
    (512, 64, "float16", "tiny_small"),
    (128, 128, "float16", "tiny"),
    (256, 256, "float16", "small"),
    (512, 512, "float16", "medium_small"),
    (1024, 512, "float16", "medium_small_batch"),
    (256, 1024, "float16", "medium"),
    (512, 1024, "float16", "medium_small"),
    (1024, 1024, "float16", "medium"),
    (2048, 1024, "float16", "medium_large"),
    (4000, 1024, "float16", "large_small"),
    (8000, 1024, "float16", "large_small"),
    (128, 2048, "float16", "small_large"),
    (256, 2048, "float16", "small_large_batch"),
    (512, 2048, "float16", "medium_large"),
    (1024, 2048, "float16", "medium_large_batch"),
    (2000, 2048, "float16", "large_large"),
    (4000, 2048, "float16", "large_large"),
    (8000, 2048, "float16", "huge_large"),
    (128, 4096, "float16", "small_4k"),
    (256, 4096, "float16", "small_4k_batch"),
    (512, 4096, "float16", "medium_4k"),
    (1000, 4096, "float16", "large_4k"),
    (2000, 4096, "float16", "large_4k_batch"),
    (4000, 4096, "float16", "huge_4k"),
    (128, 8192, "float16", "small_8k"),
    (256, 8192, "float16", "small_8k_batch"),
    (512, 8192, "float16", "medium_8k"),
    (1024, 8192, "float16", "large_8k"),
    (64, 10240, "float16", "small_10k"),
    (128, 10240, "float16", "small_10k_batch"),
]

print(f"\n{'=' * 80}")
print("LayerNorm Performance Comparison (Multiple Scales)")
print(f"{'=' * 80}\n")

results = []

for batch_size, hd, dtype, name in configs:
    n_iter = 50  # Use fewer iterations for faster testing

    # 准备数据
    m = paddle.nn.LayerNorm(hd)
    m.weight.set_value(paddle.randn_like(m.weight))
    m.bias.set_value(paddle.randn_like(m.bias))

    x = paddle.rand([batch_size, hd], dtype=dtype)

    # Standard LayerNorm
    try:
        paddle.device.synchronize()
        start = time.time()
        for _ in range(n_iter):
            y = m(x)
        paddle.device.synchronize()
        std_time = time.time() - start
        std_avg = std_time / n_iter * 1000

        # Fused LayerNorm
        from paddle.incubate.nn.functional import fused_layer_norm

        paddle.device.synchronize()
        start = time.time()
        for _ in range(n_iter):
            y, _, _, _ = fused_layer_norm(x, m.weight, m.bias, m._epsilon, 1)
        paddle.device.synchronize()
        fused_time = time.time() - start
        fused_avg = fused_time / n_iter * 1000

        ratio = fused_avg / std_avg
        results.append((batch_size, hd, name, std_avg, fused_avg, ratio))

        if ratio < 0.9:
            status = "FASTER"  # Fused is faster
        elif ratio > 1.1:
            status = "SLOWER"  # Fused is slower
        else:
            status = "SIMILAR"

        print(
            f"{name:20s} {batch_size:5d}x{hd:5d}  Std:{std_avg:6.3f}ms  Vec:{fused_avg:6.3f}ms  {status} ({ratio:.2f}x)"
        )
    except Exception as e:
        print(f"{name:20s} {batch_size:5d}x{hd:5d}  ERROR: {e}")

# 总结：分析哪些配置快哪些慢
print(f"\n{'=' * 80}")
print("Summary")
print(f"{'=' * 80}\n")

# 按配置分类
faster_configs = [r for r in results if r[5] < 0.9]
slower_configs = [r for r in results if r[5] > 1.1]
similar_configs = [r for r in results if 0.9 <= r[5] <= 1.1]

print(f"Vectorized kernel FASTER: {len(faster_configs)} configs")
for r in faster_configs:
    print(f"  {r[2]:20s} {r[0]:5d}x{r[1]:5d}  {r[5]:.2f}x")

print(f"\nVectorized kernel SLOWER: {len(slower_configs)} configs")
for r in slower_configs:
    print(f"  {r[2]:20s} {r[0]:5d}x{r[1]:5d}  {r[5]:.2f}x")

print(f"\nVectorized kernel SIMILAR: {len(similar_configs)} configs")
