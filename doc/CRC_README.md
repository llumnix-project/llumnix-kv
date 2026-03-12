# CRC Verification Logic

### Data Structures
1. **`data`**: `std::vector<std::vector<IpcBlock>>`
   - Outer layer: tensor index (one vector per tensor)
   - Inner layer: list of all IpcBlocks for that tensor
   - Each IpcBlock contains: `src_offset`, `dst_offset`, `length`

2. **`layer_gdrcpy_mem_`**: `std::vector<std::unique_ptr<GdrMemDesc>>`
   - Flattened storage: `[layer0_tensor0, layer0_tensor1, layer1_tensor0, layer1_tensor1, ...]`
   - Index calculation: `layer_idx * num_tensors_per_layer + tensor_idx`

### Prefill Node CRC Calculation Flow (send_data)
1. Each layer calls `send_data(layer_idx)`
2. Iterate through all IpcBlocks of each tensor
3. Use `get_layer_cpu_ptr(ctx, layer_idx)` to get CPU pointer
4. For each IpcBlock, calculate CRC using `src_offset`: `crc32_z(crc, layer_cpu_ptr + src_offset, len)`
5. Accumulate CRC of all layers into `self.crc_`

### Prefill Node Send CRC Request (get_remote_crc)
1. Calculate total number of IpcBlocks `bodycnt`
2. Send: `lcrc` (locally calculated CRC) + `bodycnt` + all IpcBlock `(dst_offset, length)` pairs. The sent IpcBlocks are obtained from calling `parse_block` on the last layer. Currently all layers have the same block information for all models; adaptation will be needed if models with different block information per layer are encountered.

### Decode Node CRC Calculation (resp_remote_crc)
1. Receive `local_crc` and `offlencnt`
2. Receive all `(offset, length)` pairs
3. Iterate through all `layer_descs`, for each layer_desc:
   - Calculate CRC using all offset/length pairs
