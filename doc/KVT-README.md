# KVT (KV Transfer)

## 概述

KVT（KV Transfer）是一个高性能、零开销的 KV Cache 传输模块，专为 LLM 分布式推理场景设计。它负责在两个节点之间完成 KV Cache 的高效传输，支持多种模型架构和 Cache 布局。

## 设计目标

KVT 的设计源于以下核心需求：

1. **旁路设计**：不对 step 主流程做大的改动，做到旁路式集成
2. **零额外开销（Zero Overhead）**：不会因为 KV Cache 传输在 Step 执行链路引入额外负载
3. **Full CUDA Graph 兼容**：支持 CUDA Graph 优化，不引入 CPU 同步点
4. **通用性**：支持多种模型架构（FlashAttention、GDN、DSA 等）和 Cache 布局

## 核心架构

KVT 分为四个核心模块：

### 1. 接入层（Python Binding）

负责与 Python 侧对接，主要功能：

- 通过 CUDA Event 通知 layer 计算完成，实现 Full CUDA Graph 兼容
- 支持 P 节点 full-cuda-graph，D 节点无需运行任何逻辑

### 2. ParseBlock（块解析）

根据 layer 信息和请求信息，计算待发送的 `IpcBlock` 列表：

```rust
struct IpcBlock {
    src_off: usize,   // 源端偏移
    dst_off: usize,   // 目的端偏移
    len: usize,       // 传输长度
}
```

**设计演进**：

早期 KVT 假设 Cache shape 为 `(num_blocks, block_size, 2, num_heads, head_dim)`，即每个 token 的 K/V 在同一个 block 内。

但 vLLM 使用 FlashAttention 时，Cache shape 为 `(2, num_blocks, block_size, num_kv_heads, head_size)`，即 K/V 分离在不同 block。

**解决方案**：将 ParseBlock 抽离为可插拔策略：
- 初始化时 `block_size_bytes`、`token_size_bytes` 仍按"token kv 在一起"的假设计算
- ParseBlock 在解析时根据实际 layout 重新解释偏移量

这一设计使得后续支持新架构（如 Qwen3-Next GDN、DeepSeek DSA）时，只需新增对应的 ParseBlock 实现。

### 3. 控制层

负责：
- 对端连接维护
- 监听 layer 计算完成信号（CUDA Event）
- 调度传输层完成数据传输
- 传输出错后的容错处理

### 4. 传输层

负责传输 `Vec<IpcBlock>`，支持多种传输后端：

- **GPU Direct RDMA（GDR）**：GPU 显存直连，最低延迟
- **TCP**：绕开 GPU/GDR 通路，与 EP all2all 流量隔离
- **共享内存**：单机多卡场景

## 物理布局抽象

KVT 提供的传输抽象：

```
每个 layer → 一块显存
    ↓
显存 → 若干 Block（相同字节大小）
    ↓
Block → 若干 Token（相同字节大小）
```

初始化时传递的物理布局参数：
- `block_size_bytes`: 每个 block 的字节大小
- `token_size_bytes`: 每个 token 的字节大小
- `num_blocks`: block 数量


## 与 HybridConnector 的集成

KVT 与 HybridConnector 协同工作：

```
┌─────────────────────────────────────────────────────┐
│              vLLM Engine (Python)                   │
│                                                     │
│  ┌─────────────┐    ┌──────────────────────────┐    │
│  │  Scheduler  │    │   HybridConnector        │    │
│  └─────────────┘    │  ┌────────────────────┐  │    │
│                     │  │   KVT ( C++ )      │  │    |
│                     │  │  ┌──────────────┐  │  │    │
│                     │  │  │ ParseBlock   │  │  │    │
│                     │  │  └──────────────┘  │  │    │
│                     │  └────────────────────┘  │    │
│                     └──────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

**职责划分**：
- **KVT**：负责底层 KV Cache 传输
- **HybridConnector**：负责请求生命周期管理、容错、Backend 协调


## 项目状态

- ✅ FlashAttention Cache 布局支持
- ✅ Full CUDA Graph 兼容
- ✅ GDR 传输支持
- ✅ TCP 传输支持
- ✅ Qwen3-Next GDN 支持
- ✅ DeepSeek DSA 支持
