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
print("Testing with flags to enable vectorized kernel")
print(f"{'=' * 60}")

# 检查对齐条件
m = paddle.nn.LayerNorm(hd)
m.weight.set_value(paddle.randn_like(m.weight))
m.bias.set_value(paddle.randn_like(m.bias))

x = paddle.rand([batch_size, hd], dtype="float16")

# 检查内存对齐
print(f"x addr: 0x{x._ptr()}")
print(f"x addr % 16: {x._ptr() % 16}")
print(f"hd % 4: {hd % 4}")
print(f"hd >= 128: {hd >= 128}")
print(f"weight addr: 0x{m.weight._ptr()}")
print(f"weight addr % 16: {m.weight._ptr() % 16}")
print(f"bias addr: 0x{m.bias._ptr()}")
print(f"bias addr % 16: {m.bias._ptr() % 16}")

# 测试
y = m(x)
print("\nFirst test passed!")

# 测试性能
paddle.device.synchronize()
start = time.time()
for _ in range(n_iter):
    y = m(x)
paddle.device.synchronize()
std_time = time.time() - start
print(f"Standard: {std_time:.4f}s, avg: {std_time / n_iter * 1000:.4f}ms")
