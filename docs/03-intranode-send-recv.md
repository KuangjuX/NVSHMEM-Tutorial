# 节点内 Send/Recv 实现详解

本文档详细解释节点内（intranode）send/recv 操作的正确实现方式，以及相关的同步机制。

## 核心概念

### 1. 内存模型
节点内通信基于 **CUDA IPC (Inter-Process Communication)** 和 **NVLink** 互连：

- 每个进程分配一块对称的 GPU 内存 buffer
- 通过 CUDA IPC，每个进程都能访问其他进程的 buffer
- `buffer_ptrs_[i]` 指向第 i 个进程的 buffer
- `barrier_signal_ptrs_[i][j]` 用于进程间信号同步

### 2. 通信模式
```
发送方 (Rank A)                    接收方 (Rank B)
    |                                   |
    |------ 数据写入 B 的 buffer ------>|
    |                                   |
    |------ 设置 "数据就绪" 信号 ------>|
    |                                   |
    |<----- 等待 "接收完成" 确认 -------|
```

## 实现细节

### 发送方 (intranode_send)

```cpp
void Buffer::intranode_send(const torch::Tensor& tensor, int rank) {
    int target_nvl_rank = rank % num_nvl_ranks_;
    
    // Step 1: 将数据直接写入接收方的 buffer
    // 关键：写入 buffer_ptrs_[target_nvl_rank]，不是自己的 buffer
    CUDA_CHECK(cudaMemcpyAsync(buffer_ptrs_[target_nvl_rank], tensor.data_ptr(), 
                               tensor.nbytes(), cudaMemcpyDeviceToDevice,
                               comm_stream_));

    // Step 2: 通知接收方数据已准备好
    // 在接收方的信号数组中设置信号
    int* signal_ptr = &barrier_signal_ptrs_[target_nvl_rank][nvl_rank_];
    intranode_sync::launch_signal_completion(signal_ptr, 1, comm_stream_);

    // Step 3: 等待接收方确认
    int* ack_ptr = &barrier_signal_ptrs_[nvl_rank_][target_nvl_rank];
    intranode_sync::launch_wait_for_signal(ack_ptr, 1, comm_stream_);

    // Step 4: 清理信号
    intranode_sync::launch_reset_signal(ack_ptr, comm_stream_);
    cudaStreamSynchronize(comm_stream_);
}
```

### 接收方 (intranode_recv)

```cpp
void Buffer::intranode_recv(torch::Tensor& tensor, int rank) {
    int source_nvl_rank = rank % num_nvl_ranks_;
    
    // Step 1: 等待发送方的数据准备信号
    int* ready_ptr = &barrier_signal_ptrs_[nvl_rank_][source_nvl_rank];
    intranode_sync::launch_wait_for_signal(ready_ptr, 1, comm_stream_);
    
    // Step 2: 从自己的 buffer 读取数据
    // 关键：从 buffer_ptrs_[nvl_rank_] 读取，发送方已写入数据
    CUDA_CHECK(cudaMemcpyAsync(tensor.data_ptr(), buffer_ptrs_[nvl_rank_],
                               tensor.nbytes(), cudaMemcpyDeviceToDevice,
                               comm_stream_));

    // Step 3: 确认接收完成
    int* ack_ptr = &barrier_signal_ptrs_[source_nvl_rank][nvl_rank_];
    intranode_sync::launch_signal_completion(ack_ptr, 1, comm_stream_);

    // Step 4: 清理信号
    intranode_sync::launch_reset_signal(ready_ptr, comm_stream_);
    cudaStreamSynchronize(comm_stream_);
}
```

## 同步机制

### GPU 端原子操作
使用 GPU kernel 进行高效的原子操作同步：

```cpp
// 设置信号
__global__ void signal_completion(int* signal_ptr, int value) {
    atomicExch(signal_ptr, value);
}

// 等待信号
__global__ void wait_for_signal(int* signal_ptr, int expected_value) {
    while (atomicAdd(signal_ptr, 0) != expected_value) {
        __nanosleep(100);  // 短暂休眠减少内存压力
    }
}
```

### 信号数组设计
```cpp
// barrier_signal_ptrs_[i][j] 的含义：
// - i: 信号数组的所有者（哪个进程的信号数组）
// - j: 信号的来源（来自哪个进程的信号）

// 例如：barrier_signal_ptrs_[1][0] 表示：
// - 进程 1 的信号数组中
// - 来自进程 0 的信号
```

## 错误的实现方式

### 常见错误 1：内存访问模式错误
```cpp
// 错误的发送实现
void wrong_send(const torch::Tensor& tensor, int rank) {
    // 错误：写入自己的 buffer
    cudaMemcpy(buffer_ptrs_[nvl_rank_], tensor.data_ptr(), ...);
}

// 错误的接收实现  
void wrong_recv(torch::Tensor& tensor, int rank) {
    // 错误：从发送方的 buffer 读取
    cudaMemcpy(tensor.data_ptr(), buffer_ptrs_[source_rank], ...);
}
```

### 常见错误 2：缺乏同步
```cpp
// 错误：没有同步机制
void wrong_send_no_sync(const torch::Tensor& tensor, int rank) {
    cudaMemcpy(buffer_ptrs_[target_rank], tensor.data_ptr(), ...);
    // 发送方立即返回，接收方可能还没开始接收
}
```

## 性能优化

### 1. 异步操作
- 使用 `cudaMemcpyAsync` 而不是同步版本
- 在专用的通信流上执行操作
- GPU 端信号同步避免 CPU-GPU 同步开销

### 2. 内存对齐
```cpp
// 确保 buffer 对齐以获得最佳性能
void* aligned_buffer = nvshmem_align(alignment, size);
```

### 3. 批量操作
```cpp
// 对于小消息，考虑批量发送以减少同步开销
void batch_send(const std::vector<torch::Tensor>& tensors, int rank);
```

## 使用示例

### Python 接口使用
```python
from nvshmem_tutorial import NvshmemBuffer

# 初始化 buffer
buffer = NvshmemBuffer(group, rank, world_size, nvl_bytes, rdma_bytes)

# 发送数据
if rank == 0:
    tensor = torch.ones(1024, dtype=torch.uint8, device="cuda")
    buffer.send(tensor, 1)  # 发送给 rank 1

# 接收数据
if rank == 1:
    tensor = torch.zeros(1024, dtype=torch.uint8, device="cuda")
    buffer.recv(tensor, 0)  # 从 rank 0 接收
```

### 基准测试中的应用
```python
def benchmark_intranode_send_recv(rank, world_size, size_bytes):
    buffer = NvshmemBuffer(dist.group.WORLD, rank, world_size, 
                          1024*1024*1024, 0)
    
    if rank == 0:
        tensor = torch.ones(size_bytes, dtype=torch.uint8, device="cuda")
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(num_trials):
            buffer.send(tensor, 1)
        end_event.record()
        
        torch.cuda.synchronize()
        time_ms = start_event.elapsed_time(end_event)
        bandwidth = (size_bytes * num_trials) / (time_ms / 1000.0) / 1e9
        
    elif rank == 1:
        tensor = torch.zeros(size_bytes, dtype=torch.uint8, device="cuda")
        for _ in range(num_trials):
            buffer.recv(tensor, 0)
```

## 调试技巧

### 1. 验证内存访问
```cpp
// 添加调试输出确认内存访问模式
printf("Rank %d: Writing to buffer_ptrs_[%d] = %p\n", 
       nvl_rank_, target_nvl_rank, buffer_ptrs_[target_nvl_rank]);
```

### 2. 信号状态检查
```cpp
// 检查信号状态
printf("Signal state: barrier_signal_ptrs_[%d][%d] = %d\n", 
       target_rank, source_rank, barrier_signal_ptrs_[target_rank][source_rank]);
```

### 3. 超时检测
```cpp
// 添加超时机制避免死锁
auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
while (*signal_ptr != expected_value) {
    if (std::chrono::steady_clock::now() > timeout) {
        throw std::runtime_error("Send/Recv timeout");
    }
    std::this_thread::sleep_for(std::chrono::microseconds(10));
}
```

## 总结

正确的节点内 send/recv 实现需要：

1. **正确的内存访问模式**：发送方写入接收方的 buffer，接收方从自己的 buffer 读取
2. **可靠的同步机制**：使用原子操作确保操作顺序
3. **高效的 GPU 端实现**：避免 CPU-GPU 同步开销
4. **适当的错误处理**：检查 buffer 容量、超时等

这种实现方式能够充分利用 NVLink 的高带宽和低延迟特性，为节点内通信提供最佳性能。
