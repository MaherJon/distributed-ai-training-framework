// MAHE 局域网发现：Master UDP 广播 beacon，Worker 监听后主动入网
// beacon 负载（文本）：MAHE1|<cluster_name>|<tcp_port>|<token_sha256_hex前16>
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>

#include "mahe/transport.hpp"

namespace mahe {

struct BeaconInfo {
  std::string cluster_name;
  uint16_t tcp_port{0};
  std::string token_fingerprint; // 仅指纹，不广播明文 token
  std::string from_addr;
};

class BeaconBroadcaster {
public:
  // interval_ms 周期广播
  bool start(const std::string& cluster_name, uint16_t tcp_port,
             const std::string& token_fingerprint, uint16_t udp_port,
             int interval_ms = 2000);
  void stop();
  ~BeaconBroadcaster();

private:
  void loop(int interval_ms);
  sock::socket_t fd_{sock::INVALID};
  uint16_t udp_port_{0};
  std::string payload_;
  std::thread thread_;
  std::atomic<bool> running_{false};
};

class BeaconListener {
public:
  using BeaconHandler = std::function<void(const BeaconInfo&)>;

  // 监听 udp_port，过滤魔数；收到合法 beacon 回调（可能重复，调用方去重）
  bool start(uint16_t udp_port, BeaconHandler on_beacon);
  void stop();
  ~BeaconListener();

private:
  void loop();
  sock::socket_t fd_{sock::INVALID};
  std::thread thread_;
  std::atomic<bool> running_{false};
  BeaconHandler on_beacon_;
};

// token 指纹（SHA-256 前 8 字节的十六进制；此处用 FNV-1a 简化实现，
// 仅用于广播比对，不构成安全边界——安全校验在 TCP 握手中做）
std::string token_fingerprint(const std::string& token);

} // namespace mahe
