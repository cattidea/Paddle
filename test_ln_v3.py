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

"""
Test fast_ln_v3 kernel for LayerNorm
"""

import time

import paddle

# Configure
hidden_size = 2048
batch_size = 128
n_iter = 100
dtype = "float16"  # Use float16 for vectorized kernel

print(f"\n{'=' * 60}")
print("Testing LayerNorm optimizations")
print(f"{'=' * 60}")
print(f"Hidden size: {hidden_size}, Batch size: {batch_size}")
print(f"Dtype: {dtype}, Iterations: {n_iter}")
print(f"{'=' * 60}\n")

# Prepare data
m_std = paddle.nn.LayerNorm(hidden_size)
m_std.weight.set_value(paddle.randn_like(m_std.weight))
m_std.bias.set_value(paddle.randn_like(m_std.bias))

x = paddle.rand([batch_size, hidden_size], dtype=dtype)

# Test 1: Standard LayerNorm
print("Testing Standard LayerNorm...")
y_std = m_std(x)
paddle.device.synchronize()
start = time.time()
for _ in range(n_iter):
    y = m_std(x)
paddle.device.synchronize()
std_time = time.time() - start
print(f"  Standard: {std_time:.4f}s, avg: {std_time / n_iter * 1000:.4f}ms")

# Test 2: Fused LayerNorm
from paddle.incubate.nn.functional import fused_layer_norm

print("\nTesting Fused LayerNorm (OneFlow)...")
y_fused, _, _, _ = fused_layer_norm(
    x, m_std.weight, m_std.bias, m_std._epsilon, 1
)
paddle.device.synchronize()
start = time.time()
for _ in range(n_iter):
    y, _, _, _ = fused_layer_norm(
        x, m_std.weight, m_std.bias, m_std._epsilon, 1
    )
paddle.device.synchronize()
fused_time = time.time() - start
print(f"  Fused:   {fused_time:.4f}s, avg: {fused_time / n_iter * 1000:.4f}ms")
print(f"  Speedup: {std_time / fused_time:.2f}x")

# Verify correctness
max_diff = (y_std - y_fused).abs().max()
print(f"  Max diff: {max_diff:.6f}")

# Test with bf16
print("\n" + "=" * 60)
print("Testing with bfloat16...")
x_bf16 = paddle.rand([batch_size, hidden_size], dtype="bfloat16")
m_bf16 = paddle.nn.LayerNorm(hidden_size)

y_std_bf16 = m_bf16(x_bf16)
y_fused_bf16, _, _, _ = fused_layer_norm(
    x_bf16, m_bf16.weight, m_bf16.bias, m_bf16._epsilon, 1
)

paddle.device.synchronize()
start = time.time()
for _ in range(n_iter):
    y = m_bf16(x_bf16)
paddle.device.synchronize()
std_time_bf16 = time.time() - start

paddle.device.synchronize()
start = time.time()
for _ in range(n_iter):
    y, _, _, _ = fused_layer_norm(
        x_bf16, m_bf16.weight, m_bf16.bias, m_bf16._epsilon, 1
    )
paddle.device.synchronize()
fused_time_bf16 = time.time() - start

print(
    f"  Standard: {std_time_bf16:.4f}s, avg: {std_time_bf16 / n_iter * 1000:.4f}ms"
)
print(
    f"  Fused:   {fused_time_bf16:.4f}s, avg: {fused_time_bf16 / n_iter * 1000:.4f}ms"
)
print(f"  Speedup: {std_time_bf16 / fused_time_bf16:.2f}x")

max_diff_bf16 = (y_std_bf16 - y_fused_bf16).abs().max()
print(f"  Max diff: {max_diff_bf16:.6f}")

print(f"\n{'=' * 60}")
print("Test completed!")
print(f"{'=' * 60}")
