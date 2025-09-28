#include "thrid_party/logging.h"
#include <netinet/in.h>
#include <unistd.h>
#include <dlfcn.h>
#include "common.h"
#include <string.h>
#include <thread>
#include <mutex>
#include "envcfg.h"

namespace blade_llm {

typedef int mallctl_t(const char *name, void *oldp, size_t *oldlenp,
                      void *newp, size_t newlen);

// NULL means disable monitor
static mallctl_t *p_mallctl = NULL;

const char PROF_ACTIVE[] = "prof active";
const char PROF_DEACTIVE[] = "prof deactive";
const char PROF_DUMP[] = "prof dump ";

static int set_prof_active(bool newval) {
  bool old_val = false;
  size_t old_len = sizeof(old_val);
  if (!p_mallctl) {
    return -33;
  }
  int jeret = p_mallctl("prof.active", &old_val, &old_len, &newval, sizeof(newval));
  if (jeret) {
    LOG(ERROR) << "mallctl err=" << jeret << " errno=" << errno;
    return -34;
  }
  return old_val;
}

static char* getfilepath(char* cmd) {
  char* cmd_bak = cmd;
  while (*cmd != '\0' && !isspace(*cmd)) {
    ++cmd;
  }
  *cmd = '\0';
  return cmd_bak;
}

static void conn_cmd(int conn_fd, char* cmd, size_t cmdlen) {
  assert(cmd[cmdlen] == '\0');
  LOG(INFO) << "conn cmd:fd=" << conn_fd << " cmd=" << cmd;

  if (strncmp(cmd, PROF_ACTIVE, sizeof(PROF_ACTIVE) - 1) == 0) {
    auto preval = set_prof_active(true);
    LOG(INFO) << "prof active: prev=" << preval;
    dprintf(conn_fd, "prev=%d\n", preval);
    return;
  }

  if (strncmp(cmd, PROF_DEACTIVE, sizeof(PROF_DEACTIVE) - 1) == 0) {
    auto preval = set_prof_active(false);
    LOG(INFO) << "prof deactive: prev=" << preval;
    dprintf(conn_fd, "prev=%d\n", preval);
    return;
  }

  constexpr auto DUMPLEN = sizeof(PROF_DUMP) - 1;
  if (strncmp(cmd, PROF_DUMP, DUMPLEN) == 0) {
    auto* filepath = getfilepath(&cmd[DUMPLEN]);
    if (*filepath == '\0') {
      LOG(ERROR) << "prof dump: empty filepath";
      dprintf(conn_fd, "empty filepath\n");
      return;
    }
    int jeret = p_mallctl("prof.dump", NULL, NULL, &filepath, sizeof(filepath));
    if (jeret) {
      LOG(ERROR) << "prof dump: err=" << jeret << " errno=" << errno;
      dprintf(conn_fd, "error\n");
      return;
    }
    LOG(INFO) << "prof dump: file=" << filepath;
    dprintf(conn_fd, "ok\n");
    return;
  }

  return;
}

static void conn_main(int conn_fd) {
  char buf[1024];
  ssize_t n;
  while ((n = read(conn_fd, buf, sizeof(buf) - 1)) > 0) {
    buf[n] = '\0';
    conn_cmd(conn_fd, buf, n);
  }
  LOG(INFO) << "conn exit: conn_fd=" << conn_fd << " n=" << n << " errno=" << errno;
  return;
}

static void do_thdmain() {
  int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
  RTCHECK(listen_fd >= 0);
  auto _guard = FdGuard(listen_fd);

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(0);

  int sysret = bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr));
  RTCHECK_EQ(sysret, 0);
  sysret = listen(listen_fd, 7);
  RTCHECK_EQ(sysret, 0);

  socklen_t len = sizeof(addr);
  sysret = getsockname(listen_fd, (struct sockaddr*)&addr, &len);
  RTCHECK_EQ(sysret, 0);
  auto port = ntohs(addr.sin_port);
  LOG(INFO) << "monitor listen=0.0.0.0:" << port << " fd=" << listen_fd;

  while (true) {
    int conn_fd = accept(listen_fd, NULL, NULL);
    if (conn_fd < 0) {
      LOG(ERROR) << "accept error: errno=" << errno << " fd=" << listen_fd;
      usleep(1 * 1000 * 1000);
      continue;
    }
    LOG(INFO) << "accept conn=" << conn_fd << " listenfd=" << listen_fd;
    auto _guard2 = FdGuard(conn_fd);
    conn_main(conn_fd);
  }
}

static void thdmain() noexcept {
  try {
    do_thdmain();
  } catch (const std::exception& ex) {
    LOG(INFO) << "thdmain: ex=" << ex.what();
  }
  LOG(INFO) << "thdmain: exit";
}

static void do_init() {
  assert(p_mallctl == nullptr);
  p_mallctl = (mallctl_t*)dlsym(RTLD_DEFAULT, "mallctl");
  if (p_mallctl == nullptr) {
    return;
  }
  if (env_heap_prof()) {
    auto prev = set_prof_active(true);
    LOG(INFO) << "monitor: prof active. prev=" << prev;
  }
  std::thread(thdmain).detach();
  return;
}

void monitor_init() {
  static std::once_flag flag;
  try {
    std::call_once(flag, do_init);
  } catch (const std::exception& ex) {
    LOG(INFO) << "monitor init failed. ex=" << ex.what() << " mallctl=" << p_mallctl;
    p_mallctl = nullptr;
  }
  return;
}

} // namespace blade_llm {