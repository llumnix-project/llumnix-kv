#include <mutex>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>
#ifdef ENABLE_CURL
#include <curl/curl.h>
#endif
#include "utils/http_helper.h"
#include "thrid_party/logging.h"

namespace blade_llm {

#define HTTP_MAX_RESPONSE_SIZE 1UL << 20

static std::once_flag curl_init_flag;

typedef struct MessageStruct_t {
  char *payload;
  size_t size;
} MessageStruct;

static size_t get_callback(void *contents, size_t size, size_t nmemb, void *userp) {
  size_t real_size = size * nmemb;
  MessageStruct *msg = (MessageStruct *) userp;
  if (real_size >= HTTP_MAX_RESPONSE_SIZE) {
    printf("too large message size;\n");
    return 0;
  }

  void *ptr = realloc(msg->payload, msg->size + real_size + 1);
  msg->payload = (char *) ptr;
  memcpy(&(msg->payload[msg->size]), contents, real_size);
  msg->size += real_size;
  msg->payload[msg->size] = 0;
  return real_size;
}

static void curl_init_global() {
#ifdef ENABLE_CURL
  curl_global_init(CURL_GLOBAL_ALL);
#endif
}

CurlHttpClient::CurlHttpClient() {
#ifdef ENABLE_CURL
  std::call_once(curl_init_flag, curl_init_global);
#endif
}

void CurlHttpClient::post(const std::string &url, const std::string &data) {
#ifdef ENABLE_CURL
  CURL *curl = curl_easy_init();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
  CURLcode res = curl_easy_perform(curl);
  curl_easy_cleanup(curl);
  if (res != CURLE_OK) {
    LOG(ERROR) << "HTTP POST(" << url << ") fail, caused by: " << curl_easy_strerror(res);
    throw std::runtime_error(curl_easy_strerror(res));
  }
#else
  throw std::runtime_error("HTTP POST is not implemented");
#endif
}

std::optional<std::string> CurlHttpClient::get(const std::string &url) {
#ifdef ENABLE_CURL
  MessageStruct msg;
  msg.payload = (char *) malloc(1); // TODO: use memory pool;
  msg.size = 0;
  CURL *curl = curl_easy_init();
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, get_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *) &msg);
  CURLcode res = curl_easy_perform(curl);
  if (res != CURLE_OK) {
    LOG(ERROR) << "CURL GET(" << url << ") fail, caused by: " << curl_easy_strerror(res);
    free(msg.payload);
    curl_easy_cleanup(curl);
    throw std::runtime_error(curl_easy_strerror(res));
  }
  long http_code = 0;
  curl_easy_getinfo (curl, CURLINFO_RESPONSE_CODE, &http_code);
  std::string data(msg.payload);
  free(msg.payload);
  curl_easy_cleanup(curl);

  if (http_code == 200) {
    return data;
  } else if (http_code == 404) {
    return std::nullopt;
  } else {
    throw std::runtime_error("HTTP GET(" + url + ") fail, http_code: " + std::to_string(http_code));
  }
#else
  throw std::runtime_error("HTTP GET is not implemented");
#endif
}
}
