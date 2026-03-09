# Paddle LayerNorm 优化记录

## 问题背景
- 标准 LayerNorm kernel 性能较慢
- 需要参考 PyTorch 的实现进行优化
- 目标：加速大尺度（如 8000*1024）的 LayerNorm

## PyTorch LayerNorm 优化技巧 (来自 `pytorch-main/aten/src/ATen/native/cuda/layer_norm_kernel.cu`)

### 1. 向量化加载/存储
```cpp
template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
  T val[vec_size];
};

constexpr int vec_size = 4;  // 4元素向量化
```

### 2. Welford 算法
- 在线计算 mean 和 variance，数值更稳定
- 单 pass 完成，减少内存访问

### 3. Multi-warp 策略
```cpp
dim3 threads(kWarpSize, warp_count);  // 多个 warp 处理一行
```
- warp_count 需要根据 feature_size 动态调整
- 小 feature_size 用单 warp，大 feature_size 用多 warp

### 4. 单 Kernel 实现
- 计算统计量和应用归一化在一个 kernel 中完成
- 减少 kernel launch 开销和内存访问

## Paddle 已有实现

### 1. fast_ln_v1 (NVIDIA Apex)
- 支持特定 hidden_size: 768-4096
- 位置: `paddle/phi/kernels/funcs/fast_ln_v1.h`
- 条件: hidden_size % 256 == 0

### 2. fast_ln_v2 (NVIDIA Apex)
- 需要 CUDA >= 12.0
- 位置: `paddle/phi/kernels/funcs/fast_ln_v2.h`
- 条件: input_type != FLOAT32, 1024 < hidden_size <= 10240

### 3. fused_layer_norm (OneFlow)
- Python API: `paddle.incubate.nn.functional.fused_layer_norm`
- 位置: `paddle/phi/kernels/fusion/gpu/fused_layernorm_kernel.cu`
- 特点: single pass 算法，支持 Int8

### 4. GENERIC
- 通用实现，两步计算
- 位置: `paddle/phi/kernels/funcs/layer_norm_impl.cu.h`

## 已实施的优化

### 文件修改

#### 1. `paddle/phi/kernels/funcs/layer_norm_impl.cu.h`
添加了 PyTorch 风格的 vectorized kernel:
- `AlignedVecTorch<T, 4>` - 4元素向量化结构
- `WelfordDataTorch` - Welford 算法数据结构
- `cuWelfordOnlineSum` / `cuWelfordCombine` - Welford 更新和合并
- `compute_stats_torch` - 计算统计量
- `VectorizedLayerNormForward` - 向量化前向 kernel

#### 2. `paddle/phi/kernels/gpu/layer_norm_kernel.cu`
修改 dispatch 逻辑：
- float16/bfloat16 类型优先走 GENERIC 分支（使用新 kernel）
- 动态调整 warp_count:
  - feature_size <= 1024: warp_count = 1
  - 1024 < feature_size <= 2048: warp_count = 2
  - feature_size > 2048: warp_count = 4

### 性能测试结果

配置 | Standard (ms) | Fused (ms) | Ratio
-----|---------------|------------|-------
128×2048 | 0.0999 | 0.0126 | **0.13x** (7.7x加速)
8000×1024 | 0.0303 | 0.0708 | 2.33x (变慢)
1024×1024 | 0.0172 | 0.0118 | 0.69x (1.45x加速)
2048×1024 | 0.0171 | 0.0202 | 1.18x
512×2048 | 0.0170 | 0.0116 | 0.68x (1.47x加速)

### 问题与解决

#### 问题1: 大 batch 小 hidden 性能变慢
**原因**: warp_count = 4 对小 hidden 不合适
**解决**: 动态调整 warp_count

#### 问题2: dispatch 逻辑走 FAST_LN_V1 而不是 GENERIC
**原因**: 1024 % 256 == 0，满足 FAST_LN_V1 条件
**解决**: 在 dispatch 中优先检查 float16/bfloat16

## 编译和测试

```bash
cd /root/paddlejob/tmpspace/huangzihao/Paddle/build
ninja -j128
pip install python/dist/paddlepaddle_gpu-*.whl --force-reinstall --no-deps
export PYTHONPATH=/root/paddlejob/tmpspace/huangzihao/Paddle/build/python
python test_large_scale.py
```

## 后续优化方向

1. 根据 PyTorch 的 `num_threads()` 函数动态调整 warp_count
2. 添加 alignment 检查，只在内存对齐时使用向量化
3. 考虑使用 CUB 的 `BlockReduce` 替代手动 warp reduce
4. 对大 batch 大 hidden 的情况，考虑使用 cooperative kernel

## 相关文件

- PyTorch: `/root/paddlejob/tmpspace/huangzihao/pytorch-main/aten/src/ATen/native/cuda/layer_norm_kernel.cu`
- Paddle: `paddle/phi/kernels/gpu/layer_norm_kernel.cu`
- Paddle: `paddle/phi/kernels/funcs/layer_norm_impl.cu.h`
