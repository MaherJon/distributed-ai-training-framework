#include "mahe/discovery.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>

#include "mahe/transport.hpp"

#ifndef _WIN32
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#else
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

namespace mahe {

std::string token_fingerprint(const std::string& token) {
  // FNV-1a 64bit（仅用于 beacon 展示/比对，非安全用途）
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : token) {
    h ^= c;
    h *= 1099511628211ull;
  }
  char buf[17];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return buf;
}

// ---------- BeaconBroadcaster ----------

BeaconBroadcaster::~BeaconBroadcaster() { stop(); }

bool BeaconBroadcaster::start(const std::string& cluster_name,
                              uint16_t tcp_port,
                              const std::string& token_fp, uint16_t udp_port,
                              int interval_ms) {
  sock::startup();
  fd_ = static_cast<sock::socket_t>(::socket(AF_INET, SOCK_DGRAM, 0));
  if (fd_ == sock::INVALID)
    return false;
  int flag = 1;
  setsockopt(sock::native(fd_), SOL_SOCKET, SO_BROADCAST,
             reinterpret_cast<const char*>(&flag), sizeof(flag));
  udp_port_ = udp_port;
  payload_ = "MAHE1|" + cluster_name + "|" + std::to_string(tcp_port) + "|" +
             token_fp;
  running_ = true;
  thread_ = std::thread([this, interval_ms] { loop(interval_ms); });
  return true;
}

void BeaconBroadcaster::loop(int interval_ms) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(udp_port_);
  addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);
  while (running_.load()) {
    ::sendto(sock::native(fd_), payload_.data(),
             static_cast<int>(payload_.size()), 0,
             reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    for (int i = 0; i < interval_ms / 50 && running_.load(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

void BeaconBroadcaster::stop() {
  running_ = false;
  if (thread_.joinable())
    thread_.join();
  if (fd_ != sock::INVALID) {
    sock::close(fd_);
    fd_ = sock::INVALID;
  }
}

// ---------- BeaconListener ----------

BeaconListener::~BeaconListener() { stop(); }

bool BeaconListener::start(uint16_t udp_port, BeaconHandler on_beacon) {
  sock::startup();
  on_beacon_ = std::move(on_beacon);
  fd_ = static_cast<sock::socket_t>(::socket(AF_INET, SOCK_DGRAM, 0));
  if (fd_ == sock::INVALID)
    return false;
  int flag = 1;
  setsockopt(sock::native(fd_), SOL_SOCKET, SO_REUSEADDR,
             reinterpret_cast<const char*>(&flag), sizeof(flag));
#ifdef SO_REUSEPORT
  setsockopt(sock::native(fd_), SOL_SOCKET, SO_REUSEPORT,
             reinterpret_cast<const char*>(&flag), sizeof(flag));
#endif
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(udp_port);
  if (::bind(sock::native(fd_), reinterpret_cast<sockaddr*>(&addr),
             sizeof(addr)) != 0) {
    sock::close(fd_);
    fd_ = sock::INVALID;
    return false;
  }
  running_ = true;
  thread_ = std::thread([this] { loop(); });
  return true;
}

void BeaconListener::loop() {
  std::array<char, 512> buf{};
  while (running_.load()) {
    sockaddr_in from{};
    socklen_t len = sizeof(from);
#ifdef _WIN32
    int r = ::recvfrom(sock::native(fd_), buf.data(),
                       static_cast<int>(buf.size()), 0,
                       reinterpret_cast<sockaddr*>(&from), &len);
#else
    ssize_t r = ::recvfrom(fd_, buf.data(), buf.size(), 0,
                           reinterpret_cast<sockaddr*>(&from), &len);
#endif
    if (r <= 0) {
      if (running_.load())
        continue;
      break;
    }
    std::string payload(buf.data(), static_cast<size_t>(r));
    if (payload.rfind("MAHE1|", 0) != 0)
      continue;
    // 解析 MAHE1|name|port|fp
    BeaconInfo info;
    size_t p1 = payload.find('|', 6);
    if (p1 == std::string::npos)
      continue;
    size_t p2 = payload.find('|', p1 + 1);
    if (p2 == std::string::npos)
      continue;
    info.cluster_name = payload.substr(6, p1 - 6);
    info.tcp_port = static_cast<uint16_t>(
        std::stoi(payload.substr(p1 + 1, p2 - p1 - 1)));
    info.token_fingerprint = payload.substr(p2 + 1);
    char ip[INET_ADDRSTRLEN] = {0};
    inet_ntop(AF_INET, &from.sin_addr, ip, sizeof(ip));
    info.from_addr = ip;
    if (on_beacon_)
      on_beacon_(info);
  }
}

void BeaconListener::stop() {
  running_ = false;
  if (fd_ != sock::INVALID) {
#ifdef _WIN32
    shutdown(sock::native(fd_), SD_BOTH);
#else
    ::shutdown(fd_, SHUT_RDWR);
#endif
    sock::close(fd_);
    fd_ = sock::INVALID;
  }
  if (thread_.joinable())
    thread_.join();
}

} // namespace mahe
