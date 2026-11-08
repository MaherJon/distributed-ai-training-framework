// MAHE TCP 传输层：帧收发、连接管理、写队列背压
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "mahe/protocol.hpp"

namespace mahe {

// 跨平台套接字兼容层（POSIX / Winsock）
namespace sock {
#ifdef _WIN32
using socket_t = uintptr_t; // SOCKET
inline constexpr socket_t INVALID = static_cast<socket_t>(~0ull);
// SOCKET 是 64 位句柄；传给 Winsock API 时必须保持原宽度
using native_t = uintptr_t;
inline native_t native(socket_t s) { return s; }
#else
using socket_t = int;
inline constexpr socket_t INVALID = -1;
using native_t = int;
inline native_t native(socket_t s) { return s; }
#endif

bool startup();  // Winsock 初始化（POSIX 下空操作）
void cleanup();
void close(socket_t s);
bool set_nodelay(socket_t s);
bool set_reuseaddr(socket_t s);
std::string last_error();
} // namespace sock

class Connection : public std::enable_shared_from_this<Connection> {
public:
  using MessageHandler = std::function<void(const Message&)>;
  using CloseHandler = std::function<void()>;

  Connection(sock::socket_t fd, std::string peer);
  ~Connection();

  Connection(const Connection&) = delete;
  Connection& operator=(const Connection&) = delete;

  void start(MessageHandler on_msg, CloseHandler on_close);
  bool send(const Message& msg);       // 入写队列，写满返回 false（背压）
  void close();
  bool is_open() const { return open_.load(); }
  const std::string& peer() const { return peer_; }

  static constexpr size_t WRITE_QUEUE_HIGH_WATER = 256; // 背压水位

private:
  void read_loop_impl(const MessageHandler& on_msg);
  void read_loop_with_handlers(MessageHandler on_msg, CloseHandler on_close);
  void write_loop();

  sock::socket_t fd_;
  std::string peer_;
  std::atomic<bool> open_{false};
  std::atomic<bool> closing_{false};
  std::atomic<bool> fd_closed_{false};

  std::thread reader_;
  std::thread writer_;
  std::mutex wmtx_;
  std::condition_variable wcv_;
  std::queue<std::string> wqueue_;
};

class TcpServer {
public:
  using ConnectionHandler = std::function<void(std::shared_ptr<Connection>)>;

  TcpServer() = default;
  ~TcpServer();

  bool listen(uint16_t port, ConnectionHandler on_conn);
  void stop();
  uint16_t port() const { return port_; }

private:
  void accept_loop();
  sock::socket_t listen_fd_ = sock::INVALID;
  std::thread accept_thread_;
  std::atomic<bool> running_{false};
  ConnectionHandler on_conn_;
  uint16_t port_{0};
};

// 同步连接（带超时，单位毫秒；0 = 阻塞）
std::shared_ptr<Connection> tcp_connect(const std::string& host, uint16_t port,
                                        int timeout_ms = 5000);

} // namespace mahe
