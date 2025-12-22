# CRC 校验逻辑

### 数据结构
1. **`data`**: `std::vector<std::vector<IpcBlock>>`
   - 外层：tensor索引（每个tensor一个vector）
   - 内层：该tensor的所有IpcBlock列表
   - 每个IpcBlock包含：`src_offset`, `dst_offset`, `length`

2. **`layer_gdrcpy_mem_`**: `std::vector<std::unique_ptr<GdrMemDesc>>`
   - 扁平化存储：`[layer0_tensor0, layer0_tensor1, layer1_tensor0, layer1_tensor1, ...]`
   - 索引方式：`layer_idx * num_tensors_per_layer + tensor_idx`

### Prefill node CRC计算流程（send_data）
1. 每个layer调用 `send_data(layer_idx)`
2. 遍历每个tensor的IpcBlock
3. 使用 `get_layer_cpu_ptr(ctx, layer_idx)` 获取CPU指针
4. 对每个IpcBlock，使用 `src_offset` 计算CRC：`crc32_z(crc, layer_cpu_ptr + src_offset, len)`
5. 累积所有layer的CRC到 `self.crc_`

### Prefill node发送CRC请求（get_remote_crc）
1. 计算所有IpcBlock的总数 `bodycnt`
2. 发送：`lcrc`（本地计算的CRC）+ `bodycnt` + 所有IpcBlock的 `(dst_offset, length)` 对，发送的IpcBlock是最后一层调用`parse_block`得到的。目前的模型所有layer的block信息是一样的，后续如果遇到不同layer有不同block信息的模型需要适配。

### Decode node CRC计算（resp_remote_crc）
1. 接收 `local_crc` 和 `offlencnt`
2. 接收所有 `(offset, length)` 对
3. 遍历所有 `layer_descs`，对每个layer_desc：
   - 使用所有offset/length对计算CRC
