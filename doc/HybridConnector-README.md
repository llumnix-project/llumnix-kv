# HybridConnector

## 概述

HybridConnector 是一个为 LLM 引擎设计的通用 KV Cache 异步传输框架，最初为 vLLM PD（Prefill-Decode）分离场景开发，现已演变为支持多种 KV Cache "搬迁" 场景的统一解决方案。

## 设计理念

### 核心思想

LLM 引擎与 KV Cache 传输的关系类似于 **Linux 内核与驱动程序**：
- LLM 引擎提供稳定、通用的核心计算能力
- KV Cache 传输高度依赖具体部署环境，应作为可插拔的"驱动"存在

基于这一理念，HybridConnector 遵循以下设计原则：

1. **零侵入**：不侵入引擎主链路，引擎无需感知 KV Cache 传输细节
2. **零额外开销**：KV Cache 未就绪的请求对 Scheduler 完全透明，不会引入 dummy step 等轮询机制
3. **极简接口**：仅提供必要的 `start_load_kv` 和 `save_kv_layer` 接口
4. **全异步**：所有 KV Cache 传输逻辑在独立线程/进程中异步完成

### 架构对比

| 传统方案 | HybridConnector |
|---------|----------------|
| Scheduler 主动感知 KV Cache 状态 | Scheduler 完全无感知 |
| 通过 dummy step 轮询更新状态 | 异步回调通知 |
| 同步接口阻塞 step | 全异步非阻塞 |
| 引入大量 PD 分离逻辑到引擎 | 引擎代码零侵入 |
| 容错支持不完善 | 完整的请求生命周期管理 |

## 核心架构

HybridConnector 由两个核心模块组成：

### 1. Connector（连接器）

Connector 负责为 Backend 提供运行环境，承担以下职责：
- 请求生命周期管理
- 动态扩缩容
- 链路容错控制
- Backend 协调调度

**关键创新：引用计数解耦机制**

对于需要传输 KV Cache 的请求 R，HybridConnector 通过复用 vLLM 的 Block 引用计数（refcnt）机制，实现了传输过程与请求生命周期的解耦：

```
传输开始前 → 增加 R 对应 KV Cache Block 的 refcnt
    ↓
异步传输中 → Block 不会被提前释放
    ↓
传输完成后 → 调用 free_block 递减 refcnt
    ↓
refcnt = 0 → Block 自动回收到 free list
```

这一机制确保了即使请求已结束但 KV Cache 仍在传输中的场景下，相关内存块也不会被提前释放。

### 2. Backend（后端）

Backend 负责具体的 KV Cache 传输、加载、存储操作。Backend 编写者只需了解：
- KV Cache 物理布局（shape、stride 等）
- 对应后端存储的协议与接口

**无需感知** vLLM Scheduler 内部细节。

Backend 通过注册 RPC 方法暴露能力：

```python
# PD 分离场景 - PBackend
rpcsrv.register_method(TRANSFER_KV_REQ, self._on_transfer_kv)
rpcsrv.register_method(PREFILL_REQ, self._on_prefill)
rpcsrv.register_method(SEND_DONE_REQ, self._on_send_done)
rpcsrv.register_method(ABORT_REQS_REQ, self._on_abort_reqs)

# 请求迁移场景 - MigrationBackend
rpcsrv.register_method(NEW_REQ_REQ, self._on_new_req)
rpcsrv.register_method(MIGRATE_TO_REQ, self._on_migrate_to)
rpcsrv.register_method(SUSPEND_REQ, self._on_suspend)
```

## 支持场景

HybridConnector 解决的核心问题是 **KV Cache 的"搬迁"**，支持以下场景：

### 1. PD 分离（Prefill-Decode Disaggregation）
- P 节点负责 Prefill，D 节点负责 Decode
- 通过 KVT 模块完成 P→D 的 KV Cache 传输
- D 节点无需运行任何逻辑，保持 full-cuda-graph 兼容

### 2. KVStore 持久化
- 在显存与共享存储之间搬迁 KV Cache
- 支持异步保存/加载，不阻塞计算

### 3. 请求迁移
- 将 KV Cache 在原节点与新节点之间搬迁
- 支持在线迁移，最小化服务中断

### 4. 多 Backend 组合
可同时运行多个 Backend 处理不同需求：
```
PBackend + DBackend + MigrationBackend + KVSBackend
```

## 请求生命周期

### 单请求模式
```
1. 请求 R 发送给 D 节点
2. DBackend 劫持 R，选择 P 节点，发送 PREFILL_REQ
3. P 节点开始 Prefill 并逐层传输 KV Cache
4. PREFILL_REQ 返回后，DBackend 将 R 放入 Scheduler
5. 调整 R.num_computed_tokens 为传输的 token 数
```

### 双请求模式（更灵活）
```
1. 请求 R 同时发送给 P、D 节点
2. P 节点立即开始 Prefill
3. DBackend 调用 TRANSFER_KV_REQ 告知 P 节点 DInfo
4. P 节点在下一个 step 开始传输 KV Cache
```

### Abort 处理
```
PBackend 收到 abort:
  → 终止 KVT 传输
  → 发送 SEND_DONE_REQ（带实际传输 token 数）
  → Connector 判断传输失败，返回错误码

DBackend 收到 abort:
  → 立即结束请求
  → 发送 ABORT_REQS_REQ 到 P 节点
  → KV Cache Block 通过 refcnt 机制延迟释放
```

## 与 KVT 的关系

KVT（KV Transfer）是贴近 HybridConnector 要求设计的 KV Cache 传输模块，负责在两个节点之间完成实际的 kvcache transfer 任务。

**关系定位**：
- KVT 是底层传输引擎
- HybridConnector 是 KVT 的异步运行环境
- HybridConnector 承担请求生命周期、链路容错等控制逻辑

详见 [KVT 介绍文档](./kvt-README.md)

## 技术优势

1. **极简接口**：仅保留必要的两个动作，移除 `wait_for_save`、`get_finished` 等冗余接口
2. **全异步设计**：EngineCore 独立线程运行 RPC Server，完全不阻塞主链路
3. **零 Scheduler 开销**：KV Cache 未就绪的请求对 Scheduler 不可见
4. **完整容错**：支持请求 abort、重试、超时等异常处理
5. **灵活扩展**：Backend 可插拔，支持多 Backend 组合运行

## 项目状态

- ✅ PD 分离生产环境部署验证
- ✅ KVStore 持久化支持
- ✅ 请求迁移支持
- ✅ 多 Backend 组合运行
- ✅ 完整请求生命周期管理
- ✅ Abort 容错处理
