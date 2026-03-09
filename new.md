1. 向量化加载/存储 (Vectorized Loads/Stores)

  constexpr int vec_size = 4;  // 4个元素打包
  struct alignas(sizeof(T) * vec_size) AlignedVec {
    T val[vec_size];
  };

  原理：一次访存读取 4 个 float16，内存带宽利用率提高 4 倍

  对比：
  传统: 每次读取 1 个 float16 (2 bytes)
  向量: 每次读取 4 个 float16 (8 bytes)

  2. Welford 算法

  传统方法（两步计算）：
  // Step 1: 计算 mean 和 variance
  mean = sum(x) / N
  variance = sum(x*x) / N - mean*mean

  // Step 2: 应用归一化
  y = (x - mean) / sqrt(variance + eps) * scale + bias

  Welford 方法（单 pass，数值稳定）：
  struct WelfordData {
      float mean;
      float m2;      // sum of squared differences
      float count;
  };

  WelfordData update(WelfordData curr, float val) {
      float delta = val - curr.mean;
      float new_count = curr.count + 1;
      float new_mean = curr.mean + delta / new_count;
      float delta2 = val - new_mean;
      return {new_mean, curr.m2 + delta * delta2, new_count};
  }

  优势：
  - 单 pass 完成，减少内存访问
  - 数值更稳定（避免大数抵消问题）
  - 可以并行计算，不需要等全部数据

  3. Warp-level Reduce（Warp Shuffle）

  // Warp 内的归约
  for (int offset = 16; offset > 0; offset /= 2) {
      val = __shfl_down_sync(0xffffffff, val, offset);
  }

  原理：warp 内的 32 个线程通过 shuffle 指令快速交换数据，不需要共享内存

  对比：
  传统 Block Reduce: 使用 shared memory，需要多次同步
  Warp Reduce: 单条指令，无同步开销

  4. Multi-warp 策略

  dim3 threads(32, 8);  // 8 warps
  dim3 blocks(batch_size);

  原理：每行数据由 8 个 warp 并行处理，每个 warp 处理一部分

  优势：
  - 更好的并行度，充分利用 GPU 的流处理器
  - 减少每个线程的工作量，减少寄存器压力

  5. 单 Kernel

  传统实现：
  RowwiseMomentsCUDAKernel<<<batch_size, 256>>>(x, mean, rstd);
  LayerNormForwardCUDAKernel<<<batch_size, 256>>>(x, mean, rstd, scale, bias, y);

  Vectorized 实现：
  VectorizedLayerNormForward<<<batch_size, (32, 8)>>>(x, scale, bias, mean, rstd, y);

  优势：
  - 减少 kernel launch 开销
  - 减少中间结果的内存读写（mean, rstd 写入后再读出）

  性能对比（512×2048 float16）

  ┌─────────────┬───────────────┬─────────────┬──────┐
  │    操作     │     传统      │ Vectorized  │ 加速 │
  ├─────────────┼───────────────┼─────────────┼──────┤
  │ 内存访问    │ 逐元素        │ 4元素向量   │ 4x   │
  ├─────────────┼───────────────┼─────────────┼──────┤
  │ 计算步骤    │ 2步           │ 1步         │ 2x   │
  ├─────────────┼───────────────┼─────────────┼──────┤
  │ Warp Reduce │ shared memory │ shuffle指令 │ ~2x  │
  ├─────────────┼───────────────┼─────────────┼──────┤
  │ 综合        │ 100%          │ ~60%        │ 1.7x │
  └─────────────┴───────────────┴─────────────┴──────┘
