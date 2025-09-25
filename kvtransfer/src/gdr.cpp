#include "utils/gdr.h"
#include <stdint.h>
#include <dlfcn.h>
#include "thrid_party/logging.h"
#include "common.h"
#include <mutex>
#include <memory>
#include <thread>

namespace blade_llm {

class GDRHandle;
// may be null
static std::unique_ptr<GDRHandle> g_handle;


// Dynamically handle dependency the GDR API library

/* Extracted from gdrapi.h (v2.1 Nov 2020) */

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    (1UL << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

struct gdr;
typedef struct gdr *gdr_t;

typedef struct gdr_mh_s {
  unsigned long h;
} gdr_mh_t;

struct gdr_info {
    uint64_t va;
    uint64_t mapped_size;
    uint32_t page_size;
    uint64_t tm_cycles;
    uint32_t cycles_per_ms;
    unsigned mapped:1;
    unsigned wc_mapping:1;
};
typedef struct gdr_info gdr_info_t;

/* End of gdrapi.h */

static gdr_t (*gdr_internal_open)(void) = nullptr;
static int (*gdr_internal_close)(gdr_t g) = nullptr;
static int (*gdr_internal_pin_buffer)(gdr_t g, unsigned long addr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *handle) = nullptr;
static int (*gdr_internal_unpin_buffer)(gdr_t g, gdr_mh_t handle) = nullptr;
static int (*gdr_internal_get_info)(gdr_t g, gdr_mh_t handle, gdr_info_t *info) = nullptr;
static int (*gdr_internal_map)(gdr_t g, gdr_mh_t handle, void **va, size_t size) = nullptr;
static int (*gdr_internal_unmap)(gdr_t g, gdr_mh_t handle, void *va, size_t size) = nullptr;
static void (*gdr_internal_runtime_get_version)(int *major, int *minor) = nullptr;
static void (*gdr_internal_driver_get_version)(gdr_t g, int *major, int *minor) = nullptr;

static void load_sym(void* handle, const char* sym, void** out) {
  *out = dlsym(handle, sym);
  LOG(INFO) << "load sym: sym=" << sym << ";out=" << *out << ";handle=" << handle;
  return;
}

static void load_syms() {
  void* gdrsoh = dlopen("libgdrapi.so", RTLD_NOW);
  if (gdrsoh == nullptr) {
    LOG(ERROR) << "dlopen libgdrapi.so failed";
    return;
  }
  load_sym(gdrsoh, "gdr_open", (void**)&gdr_internal_open);
  load_sym(gdrsoh, "gdr_close", (void**)&gdr_internal_close);
  load_sym(gdrsoh, "gdr_pin_buffer", (void**)&gdr_internal_pin_buffer);
  load_sym(gdrsoh, "gdr_unpin_buffer", (void**)&gdr_internal_unpin_buffer);
  load_sym(gdrsoh, "gdr_get_info", (void**)&gdr_internal_get_info);
  load_sym(gdrsoh, "gdr_map", (void**)&gdr_internal_map);
  load_sym(gdrsoh, "gdr_unmap", (void**)&gdr_internal_unmap);
  load_sym(gdrsoh, "gdr_runtime_get_version", (void**)&gdr_internal_runtime_get_version);
  load_sym(gdrsoh, "gdr_driver_get_version", (void**)&gdr_internal_driver_get_version);
  return;
}

class GDRMHandle {
  gdr_mh_t const handle_;
public:
  GDRMHandle(gdr_mh_t h) noexcept:
    handle_(h) {}

  auto handle() const noexcept {
    return this->handle_;
  }

  ~GDRMHandle();
private:
  GDRMHandle(const GDRMHandle&) = delete;
  GDRMHandle(GDRMHandle&&) = delete;
};


class MmapHandle {
  gdr_mh_t const handle_;
  void* const ptr_;
  size_t const size_;
public:
  MmapHandle(gdr_mh_t handle, void* ptr, size_t size) noexcept:
    handle_(handle), ptr_(ptr), size_(size) {}

  ~MmapHandle();

  void* ptr() const noexcept {
    return this->ptr_;
  }

private:
  MmapHandle(const MmapHandle&) = delete;
  MmapHandle(MmapHandle&&) = delete;
};


class GDRHandle {
  gdr_t const handle;
  std::mutex mtx;
public:
  explicit GDRHandle(gdr_t h) noexcept:
    handle(h) {
    LOG(INFO) << "gdr open handle=" << this->handle;
  }

  ~GDRHandle() noexcept {
    // RTCHECK(gdr_internal_close != nullptr);
    int ret = gdr_internal_close ? gdr_internal_close(this->handle) : -313131;
    LOG(INFO) << "gdr close handle=" << this->handle << ";ret=" << ret;
  }

  std::unique_ptr<GDRMHandle> pin_buffer(unsigned long addr, size_t size) {
    auto& self = *this;
    RTCHECK(gdr_internal_pin_buffer);

    gdr_mh_t ret;
    auto guard = std::lock_guard<std::mutex>(self.mtx);
    int res = gdr_internal_pin_buffer(self.handle, addr, size, 0, 0, &ret);
    LOG(INFO) << "pin buff=" << (void*)addr << ";size=" << size
              << ";mh=" << ret.h << ";gdrh=" << self.handle
              << ";res=" << res;
    RTCHECK(res == 0);
    auto mh = std::make_unique<GDRMHandle>(ret);

    gdr_info_t info;
    RTCHECK(gdr_internal_get_info);
    res = gdr_internal_get_info(self.handle, ret, &info);
    RTCHECK(res == 0);
    RTCHECK(info.va == addr && info.page_size == GPU_PAGE_SIZE);

    return mh;
  }

  std::unique_ptr<MmapHandle> map(gdr_mh_t mh, size_t size) {
    auto& self = *this;
    RTCHECK(gdr_internal_map);

    void *cpu_ptr = nullptr;
    auto guard = std::lock_guard<std::mutex>(self.mtx);
    int res = gdr_internal_map(self.handle, mh, &cpu_ptr, size);
    LOG(INFO) << "map:gdrhandle=" << self.handle << ";mh=" << mh.h
              << ";cpuptr=" << cpu_ptr << ";size=" << size << ";res=" << res;
    RTCHECK(res == 0);

    return std::make_unique<MmapHandle>(mh, cpu_ptr, size);
  }


private:
  void unmap(gdr_mh_t mh, void* cpu_ptr, size_t size) noexcept {
    auto& self = *this;

    int res = -31313131;
    if (gdr_internal_unmap) {
      auto guard = std::lock_guard<std::mutex>(self.mtx);
      res = gdr_internal_unmap(self.handle, mh, cpu_ptr, size);
    }
    LOG(INFO) << "unmap:gdrhandle=" << self.handle << ";mh=" << mh.h
              << ";cpuptr=" << cpu_ptr << ";size=" << size << ";res=" << res;

    return;
  }

  void unpin_buffer(gdr_mh_t mh) noexcept {
    auto& self = *this;

    int res = -31313131;
    if (gdr_internal_unpin_buffer) {
      auto guard = std::lock_guard<std::mutex>(self.mtx);
      res = gdr_internal_unpin_buffer(self.handle, mh);
    }
    LOG(INFO) << "unpin buffer. mh=" << mh.h << ";res=" << res << ";gdrh=" << self.handle;
    return;
  }

private:
  friend class MmapHandle;
  friend class GDRMHandle;

  GDRHandle(const GDRHandle&) = delete;
  GDRHandle(GDRHandle&&) = delete;
};

static std::unique_ptr<GDRHandle> wrap_gdr_open(void) {
  RTCHECK(gdr_internal_open != nullptr);
  auto rh = gdr_internal_open();
  RTCHECK(rh != nullptr);
  auto handle = std::make_unique<GDRHandle>(rh);

  int libmajor = 0, libminor = 0;
  RTCHECK(gdr_internal_runtime_get_version != nullptr);
  gdr_internal_runtime_get_version(&libmajor, &libminor);

  int drvmajor = 0, drvminor = 0;
  RTCHECK(gdr_internal_driver_get_version != nullptr);
  gdr_internal_driver_get_version(rh, &drvmajor, &drvminor);

  LOG(INFO) << "gdr ver: lib=" << libmajor << '.' << libminor << ";drv=" << drvmajor << '.' << drvminor;
  RTCHECK(!(libmajor < 2 || (libmajor == 2 && libminor < 1) || drvmajor < 2 || (drvmajor == 2 && drvminor < 1)));
  return handle;
}

static void init() {
  load_syms();
  g_handle = wrap_gdr_open();
  return;
}

static void gdrcpy_init() noexcept {
  static std::once_flag flag;
  try {
    std::call_once(flag, init);
  } catch (const std::exception& ex) {
    LOG(ERROR) << "gdrcpy_init init failed; ex=" << ex.what();
  }
  return;
}

GDRMHandle::~GDRMHandle() {
  g_handle->unpin_buffer(this->handle_);
}

MmapHandle::~MmapHandle() {
  auto& self = *this;
  g_handle->unmap(self.handle_, self.ptr_, self.size_);
  return;
}

GdrMemDesc::GdrMemDesc(void* cpu_ptr, std::unique_ptr<GDRMHandle> handle,
                       std::unique_ptr<MmapHandle> mmhandle) noexcept:
  cpu_ptr_(cpu_ptr),
  handle_(std::move(handle)),
  mmhandle_(std::move(mmhandle)) {}


GdrMemDesc::~GdrMemDesc() {
}

static std::unique_ptr<GdrMemDesc> do_mmap(GDRHandle* ghandle, void* gpu_ptr, size_t size) {
  assert(g_handle);
  auto const raw_ptr = uintptr_t(gpu_ptr);
  uintptr_t const aligned_gpu_ptr = raw_ptr & GPU_PAGE_MASK;
  uintptr_t const aligned_gpu_end = (raw_ptr + size + GPU_PAGE_OFFSET) & GPU_PAGE_MASK;
  size_t const aligned_size = aligned_gpu_end - aligned_gpu_ptr;

  auto mh = ghandle->pin_buffer(aligned_gpu_ptr, aligned_size);
  auto mmap_handle = ghandle->map(mh->handle(), aligned_size);
  uint8_t* aligned_cpu_ptr = (uint8_t*)mmap_handle->ptr();

  uint8_t* cpu_ptr = aligned_cpu_ptr + (raw_ptr - aligned_gpu_ptr);
  return std::make_unique<GdrMemDesc>(cpu_ptr, std::move(mh), std::move(mmap_handle));
}

std::unique_ptr<GdrMemDesc> gdrcpy_mmap(void* gpu_ptr, size_t size) noexcept {
  gdrcpy_init();
  if (!g_handle) {
    return nullptr;
  }
  try {
    return do_mmap(g_handle.get(), gpu_ptr, size);
  } catch (const std::exception& ex) {
    LOG(ERROR) << "gdrcpy_mmap failed: gpu_ptr=" << gpu_ptr << " size=" << size
               << " ex=" << ex.what() << " handle=" << g_handle.get();
    return nullptr;
  }
}

}  // namespace blade_llm {
