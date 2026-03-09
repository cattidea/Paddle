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

# 测试配置：重点测试 batch_size 临界点
configs = [
    # (batch_size, hidden_size, dtype, name)
    # batch_size <= 1024 应该用 vectorized kernel
    (128, 2048, "float16", "small_large"),
    (512, 2048, "float16", "medium_large"),
    (1024, 2048, "float16", "medium_large_batch"),
    (128, 4096, "float16", "small_4k"),
    (512, 4096, "float16", "medium_4k"),
    (1024, 4096, "float16", "large_4k"),
    # batch_size > 1024 不应该用 vectorized kernel
    (2048, 2048, "float16", "large_large"),
    (4000, 2048, "float16", "large_large"),
    (8000, 2048, "float16", "huge_large"),
    (2048, 4096, "float16", "large_4k_batch"),
    (4000, 4096, "float16", "huge_4k"),
]

print(f"\n{'=' * 80}")
print("LayerNorm Performance Test (Paddle vs Vectorized)")
print(f"{'=' * 80}\n")

for batch_size, hd, dtype, name in configs:
    n_iter = 100

    # 准备数据
    m = paddle.nn.LayerNorm(hd)
    m.weight.set_value(paddle.randn_like(m.weight))
    m.bias.set_value(paddle.randn_like(m.bias))

    x = paddle.rand([batch_size, hd], dtype=dtype)

    # 测试当前实现（应该包含 vectorized kernel）
    paddle.device.synchronize()
    start = time.time()
    for _ in range(n_iter):
        y = m(x)
    paddle.device.synchronize()
    current_time = time.time() - start
    current_avg = current_time / n_iter * 1000

    # 测试 fused_layer_norm（作为参考）
    from paddle.incubate.nn.functional import fused_layer_norm

    paddle.device.synchronize()
    start = time.time()
    for _ in range(n_iter):
        y, _, _, _ = fused_layer_norm(x, m.weight, m.bias, m._epsilon, 1)
    paddle.device.synchronize()
    fused_time = time.time() - start
    fused_avg = fused_time / n_iter * 1000

    print(
        f"{name:20s} {batch_size:5d}x{hd:5d}  Paddle:{current_avg:6.3f}ms  Fused:{fused_avg:6.3f}ms  Ratio:{current_avg / fused_avg:.2f}x"
    )
