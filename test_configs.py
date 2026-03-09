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

# 测试不同配置
configs = [
    # (batch_size, hidden_size, dtype, name)
    (128, 2048, "float16", "small batch, large hidden"),
    (8000, 1024, "float16", "large batch, small hidden"),
    (1024, 1024, "float16", "medium"),
    (2048, 1024, "float16", "medium-large"),
    (512, 2048, "float16", "small-medium"),
]

print(f"\n{'=' * 70}")
print("LayerNorm Performance Comparison")
print(f"{'=' * 70}\n")

for batch_size, hd, dtype, name in configs:
    print(f"\n[{name}] batch={batch_size}, hidden={hd}, dtype={dtype}")
    print("-" * 50)

    n_iter = 100

    # 准备数据
    m = paddle.nn.LayerNorm(hd)
    m.weight.set_value(paddle.randn_like(m.weight))
    m.bias.set_value(paddle.randn_like(m.bias))

    x = paddle.rand([batch_size, hd], dtype=dtype)

    # Standard LayerNorm
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

    print(f"  Standard: {std_avg:.4f}ms")
    print(f"  Fused:   {fused_avg:.4f}ms")
    print(f"  Ratio:   {fused_avg / std_avg:.2f}x")
