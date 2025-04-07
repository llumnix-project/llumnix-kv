#ifndef KVTRANSFER_INCLUDE_UTILS_SOCKET_HELPER_H_
#define KVTRANSFER_INCLUDE_UTILS_SOCKET_HELPER_H_
#pragma once

#include <cstdint>
#include <cstddef>

namespace blade_llm {

bool start_uds_server(const char* path, int *sock_fd);
bool try_connect_uds(const char* path, int *sock_fd);
int write_sock(int sock_fd, const void *buf, size_t buf_size);
int read_sock(int sock_fd, void *buf, size_t buf_size);
int try_read(int sock_fd, char *buf, size_t buf_size, int timeout_ms);
int wait_conn(int sock_fd, int timeout_ms);
void close_sock(int sock_fd);

} // namespace blade_llm

#endif //KVTRANSFER_INCLUDE_UTILS_SOCKET_HELPER_H_
