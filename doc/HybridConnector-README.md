# HybridConnector

## Overview

HybridConnector is a unified KV Cache asynchronous transfer framework designed for LLM engines. Initially developed for vLLM PD (Prefill-Decode) separation scenarios, it has evolved into a unified solution supporting multiple KV Cache "relocation" scenarios.

## Design Philosophy

### Core Concept

The relationship between LLM engines and KV Cache transfer is analogous to the **Linux kernel and drivers**:
- The LLM engine provides stable, generic core computing capabilities
- KV Cache transfer is highly dependent on specific deployment environments and should exist as a pluggable "driver"

Based on this concept, HybridConnector follows these design principles:

1. **Zero Intrusion**: Does not intrude into the engine's main path; the engine remains unaware of KV Cache transfer details
2. **Zero Overhead**: Requests with pending KV Cache transfers are completely transparent to the Scheduler, without introducing polling mechanisms like dummy steps
3. **Minimal Interface**: Only provides essential `start_load_kv` and `save_kv_layer` interfaces
4. **Fully Asynchronous**: All KV Cache transfer logic runs asynchronously in independent threads/processes

### Architecture Comparison

| Traditional Approach | HybridConnector |
|---------|----------------|
| Scheduler actively monitors KV Cache status | Scheduler completely unaware |
| Polling for status updates via dummy steps | Asynchronous callback notifications |
| Synchronous interfaces blocking steps | Fully asynchronous non-blocking |
| Extensive PD separation logic in engine | Zero intrusion to engine code |
| Incomplete fault tolerance support | Complete request lifecycle management |

## Core Architecture

HybridConnector consists of two core modules:

### 1. Connector

The Connector provides the runtime environment for Backends and handles:
- Request lifecycle management
- Dynamic scaling
- Link fault tolerance control
- Backend coordination and scheduling

**Key Innovation: Reference Counting Decoupling Mechanism**

For requests R requiring KV Cache transfer, HybridConnector achieves decoupling between transfer and request lifecycle by reusing vLLM's Block reference counting (refcnt) mechanism:

```
Before transfer starts → Increase refcnt of R's KV Cache Blocks
    ↓
During async transfer → Blocks are not prematurely released
    ↓
After transfer completes → Call free_block to decrement refcnt
    ↓
refcnt = 0 → Block automatically recycled to free list
```

This mechanism ensures memory blocks are not prematurely released even when the request has ended but KV Cache is still being transferred.

### 2. Backend

Backend handles specific KV Cache transfer, load, and store operations. Backend authors only need to understand:
- KV Cache physical layout (shape, stride, etc.)
- Protocols and interfaces for the corresponding backend storage

No need to be aware of vLLM Scheduler internals.

Backend exposes capabilities via RPC method registration:

```python
# PD Separation Scenario - PBackend
rpcsrv.register_method(TRANSFER_KV_REQ, self._on_transfer_kv)
rpcsrv.register_method(PREFILL_REQ, self._on_prefill)
rpcsrv.register_method(SEND_DONE_REQ, self._on_send_done)
rpcsrv.register_method(ABORT_REQS_REQ, self._on_abort_reqs)

# Request Migration Scenario - MigrationBackend
rpcsrv.register_method(NEW_REQ_REQ, self._on_new_req)
rpcsrv.register_method(MIGRATE_TO_REQ, self._on_migrate_to)
rpcsrv.register_method(SUSPEND_REQ, self._on_suspend)
```

## Supported Scenarios

HybridConnector's core problem is **KV Cache "relocation"**, supporting the following scenarios:

### 1. PD Separation (Prefill-Decode Disaggregation)
- P node handles Prefill, D node handles Decode
- KV Cache P→D transfer via KVT module
- D node requires no logic execution, maintaining full-cuda-graph compatibility

### 2. KVStore Persistence
- Relocate KV Cache between GPU memory and shared storage
- Supports async save/load without blocking computation

### 3. Request Migration
- Relocate KV Cache between original and new nodes
- Supports online migration with minimal service interruption

### 4. Multi-Backend Combination
Multiple Backends can run simultaneously for different needs:
```
PBackend + DBackend + MigrationBackend + KVSBackend
```

## Request Lifecycle

### Single Request Mode
```
1. Request R sent to D node
2. DBackend hijacks R, selects P node, sends PREFILL_REQ
3. P node starts Prefill and transfers KV Cache layer by layer
4. After PREFILL_REQ returns, DBackend places R into Scheduler
5. Adjust R.num_computed_tokens to the number of transferred tokens
```

### Dual Request Mode (More Flexible)
```
1. Request R sent to both P and D nodes
2. P node immediately starts Prefill
3. DBackend calls TRANSFER_KV_REQ to inform P node of DInfo
4. P node starts KV Cache transfer on next step
```

### Abort Handling
```
PBackend receives abort:
  → Terminate KVT transfer
  → Send SEND_DONE_REQ (with actual transferred token count)
  → Connector determines transfer failure, returns error code

DBackend receives abort:
  → Immediately end request
  → Send ABORT_REQS_REQ to P node
  → KV Cache Blocks released via refcnt mechanism with delayed release
```

## Relationship with KVT

KVT (KV Transfer) is a KV Cache transfer module designed according to HybridConnector requirements, responsible for actual KV Cache transfer between two nodes.

**Relationship**:
- KVT is the low-level transfer engine
- HybridConnector provides KVT's async runtime environment
- HybridConnector handles control logic like request lifecycle and link fault tolerance

See [KVT documentation](./kvt-README.md) for details.

## Technical Advantages

1. **Minimal Interface**: Only two essential actions retained; redundant interfaces like `wait_for_save`, `get_finished` removed
2. **Fully Asynchronous**: EngineCore runs RPC Server in independent thread, never blocking main path
3. **Zero Scheduler Overhead**: Requests with pending KV Cache are invisible to Scheduler
4. **Complete Fault Tolerance**: Supports request abort, retry, timeout, and other exception handling
5. **Flexible Extension**: Pluggable Backends supporting multi-backend combined operation

## Project Status

- ✅ PD separation production environment deployment verified
- ✅ KVStore persistence support
- ✅ Request migration support
- ✅ Multi-backend combined operation
- ✅ Complete request lifecycle management
- ✅ Abort fault tolerance handling
