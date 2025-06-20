#include <sys/socket.h>
#include <sys/un.h>
#include <sys/select.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdexcept>
#include "utils/socket_helper.h"
#include "thrid_party/logging.h"

namespace blade_llm {

bool start_uds_server(const char *path, int *sock_fd) {
  *sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (*sock_fd == -1) {
    LOG(ERROR) << "KVT socket server: fail to create server socket, errorno = " << errno;
    throw std::runtime_error("CudaIpcServer: fail to create server socket;");
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  unlink(path);
  strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
  if (bind(*sock_fd, (struct sockaddr *) &addr, sizeof(addr)) == -1) {
    LOG(ERROR) << "KVT socket server: fail to bind socket , errorno = " << errno;
    throw std::runtime_error("CudaIpcServer: fail to bind server socket");
  }

  if (listen(*sock_fd, 5) == -1) {
    LOG(ERROR) << "KVT socket server: fail to listen server socket , errorno = " << errno;
    throw std::runtime_error("fail to listen server socket;");
  }

  LOG(INFO) << "KVT socket server: start uds server:(" << path << ") for cuda ipc;";
  return true;
}

bool try_connect_uds(const char *path, int *sock_fd) {
  struct sockaddr_un addr;
  *sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (*sock_fd == -1) {
    return false;
  }
  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
  if (connect(*sock_fd, (struct sockaddr *) &addr, sizeof(addr)) == -1) {
    LOG(ERROR) << "KVT socket client: fail to connect socket server : " << path;
    return false;
  }
  LOG(INFO) << "KVT socket client: uds server: (" << path << ") connected. ";
  return true;
}

int write_sock(int sock_fd, const void *bufp, size_t buf_size) {
  auto* buf = reinterpret_cast<const char*>(bufp);
  size_t offset = 0;
  size_t left = buf_size;
  while (left > 0) {
    auto ret = send(sock_fd, buf + offset, left, 0);
    if (ret == -1) {
      return -1;
    }
    offset += ret;
    left -= ret;
  }
  return offset;
}

int read_sock(int sock_fd, void *bufp, size_t buf_size) {
  auto* buf = reinterpret_cast<char*>(bufp);
  size_t offset = 0;
  size_t left = buf_size;
  while (left > 0) {
    int ret = recv(sock_fd, buf + offset, left, MSG_WAITALL);
    // > However, the call may still return less data than requested
    // > if a signal is caught.
    if (ret == -1) {
      return -1;
    }
    if (ret == 0) {
      break;
    }
    assert(ret > 0);
    offset += ret;
    left -= ret;
  }
  return offset;
}

int try_read(int sock_fd, char *buf, size_t buf_size, int timeout_ms) {
  struct timeval timeout;
  timeout.tv_sec = 0;
  timeout.tv_usec = timeout_ms * 1000;
  fd_set read_fds;

  FD_ZERO(&read_fds);
  FD_SET(sock_fd, &read_fds);

  auto activity = select(sock_fd + 1, &read_fds, nullptr, nullptr, &timeout);
  if (activity < 0) {
    LOG(ERROR) << "KVT socket: fail to select read socket, errorno = " << errno;
    return -1;
  }

  if (activity > 0) {
    if (FD_ISSET(sock_fd, &read_fds)) {
      auto r = recv(sock_fd, buf, buf_size, 0);
      if (r < 0) {
        LOG(ERROR) << "KVT socket: fail to recv socket, errorno = " << errno;
        return -1;
      }
      return r;
    }
  }
  return 0;
}

int wait_conn(int sock_fd, int timeout_sec) {
  fd_set readfds;
  struct timeval timeout;
  timeout.tv_sec = timeout_sec;
  timeout.tv_usec = 0;
  struct sockaddr_in peer_addr;
  socklen_t len = sizeof(peer_addr);
  FD_ZERO(&readfds);
  FD_SET(sock_fd, &readfds);
  if (select(sock_fd + 1, &readfds, NULL, NULL, &timeout) > 0) {
    if (FD_ISSET(sock_fd, &readfds)) {
      int s = accept(sock_fd, (struct sockaddr *) &peer_addr, &len);
      LOG(INFO) << "KVT socket server: accept connection from " << inet_ntoa(peer_addr.sin_addr);
      return s;
    }
  }
  return -1;
}

void close_sock(int sock_fd) {
  shutdown(sock_fd, SHUT_RDWR);
  close(sock_fd);
};
} //  namespace blade_llm
