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

# 大尺度测试
hd = 1024
batch_size = 8000
n_iter = 100

print(f"\n{'=' * 60}")
print("Testing Large Scale LayerNorm")
print(f"{'=' * 60}")
print(f"Hidden size: {hd}, Batch size: {batch_size}")
print(f"Total elements: {batch_size * hd:,}")
print(f"{'=' * 60}\n")

# 准备数据
m = paddle.nn.LayerNorm(hd)
m.weight.set_value(paddle.randn_like(m.weight))
m.bias.set_value(paddle.randn_like(m.bias))

x = paddle.rand([batch_size, hd], dtype="float16")

# 测试 Standard LayerNorm
print("Testing Standard LayerNorm...")
y_std = m(x)
paddle.device.synchronize()
start = time.time()
for _ in range(n_iter):
    y = m(x)
paddle.device.synchronize()
std_time = time.time() - start
print(
    f"  Standard: {std_time:.4f}s, avg: {std_time / n_iter * 1000:.4f}ms, throughput: {batch_size * hd * n_iter / std_time / 1e9:.2f} GB/s"
)

# 测试 Fused LayerNorm
from paddle.incubate.nn.functional import fused_layer_norm

print("\nTesting Fused LayerNorm...")
y_fused, _, _, _ = fused_layer_norm(x, m.weight, m.bias, m._epsilon, 1)
paddle.device.synchronize()
start = time.time()
for _ in range(n_iter):
    y, _, _, _ = fused_layer_norm(x, m.weight, m.bias, m._epsilon, 1)
paddle.device.synchronize()
fused_time = time.time() - start
print(
    f"  Fused:   {fused_time:.4f}s, avg: {fused_time / n_iter * 1000:.4f}ms, throughput: {batch_size * hd * n_iter / fused_time / 1e9:.2f} GB/s"
)
print(f"  Speedup: {std_time / fused_time:.2f}x")

# 验证正确性
max_diff = (y_std - y_fused).abs().max()
print(f"  Max diff: {max_diff:.6f}")
