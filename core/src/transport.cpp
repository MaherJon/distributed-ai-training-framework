#include "mahe/transport.hpp"

#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#if defined(_MSC_VER)
// 仅 MSVC 支持 pragma 自动引入库；MinGW 由 build.py 传入 -lws2_32
#pragma comment(lib, "ws2_32.lib")
#endif
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace mahe {

namespace sock {

bool startup() {
#ifdef _WIN32
  WSADATA wsa;
  return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
#else
  return true;
#endif
}

void cleanup() {
#ifdef _WIN32
  WSACleanup();
#endif
}

void close(socket_t s) {
#ifdef _WIN32
  closesocket(static_cast<SOCKET>(s));
#else
  ::close(s);
#endif
}

bool set_nodelay(socket_t s) {
  int flag = 1;
  return setsockopt(native(s), IPPROTO_TCP, TCP_NODELAY,
                    reinterpret_cast<const char*>(&flag), sizeof(flag)) == 0;
}

bool set_reuseaddr(socket_t s) {
  int flag = 1;
  return setsockopt(native(s), SOL_SOCKET, SO_REUSEADDR,
                    reinterpret_cast<const char*>(&flag), sizeof(flag)) == 0;
}

std::string last_error() {
#ifdef _WIN32
  return "wsa=" + std::to_string(WSAGetLastError());
#else
  return std::strerror(errno);
#endif
}

} // namespace sock

// ---------- Connection ----------

Connection::Connection(sock::socket_t fd, std::string peer)
    : fd_(fd), peer_(std::move(peer)) {
  sock::set_nodelay(fd_);
}

Connection::~Connection() { close(); }

void Connection::start(MessageHandler on_msg, CloseHandler on_close) {
  open_ = true;
  reader_ = std::thread([this, on_msg = std::move(on_msg),
                         on_close = std::move(on_close)]() mutable {
    read_loop_with_handlers(std::move(on_msg), std::move(on_close));
  });
  writer_ = std::thread([this] { write_loop(); });
}

// 声明于下：将 read_loop 与回调封装，避免部分捕获 shared_from_this 生命周期问题
void Connection::read_loop_with_handlers(MessageHandler on_msg,
                                         CloseHandler on_close) {
  auto self = shared_from_this();
  read_loop_impl(std::move(on_msg));
  open_ = false;
  // 唤醒写线程退出
  {
    std::lock_guard<std::mutex> lk(wmtx_);
    closing_ = true;
  }
  wcv_.notify_all();
  if (on_close)
    on_close();
}

void Connection::read_loop_impl(const MessageHandler& on_msg) {
  char header[HEADER_SIZE];
  auto read_exact = [&](char* buf, size_t n) -> bool {
    size_t got = 0;
    while (got < n) {
#ifdef _WIN32
      int r = recv(static_cast<SOCKET>(fd_), buf + got,
                   static_cast<int>(n - got), 0);
#else
      ssize_t r = ::recv(fd_, buf + got, n - got, 0);
#endif
      if (r <= 0)
        return false;
      got += static_cast<size_t>(r);
    }
    return true;
  };

  try {
    while (!closing_.load()) {
      if (!read_exact(header, HEADER_SIZE))
        break;
      FrameHeader h = parse_header(header);
      Message msg;
      msg.type = h.type;
      msg.flags = h.flags;
      msg.request_id = h.request_id;
      if (h.payload_len > 0) {
        msg.payload.resize(h.payload_len);
        if (!read_exact(msg.payload.data(), h.payload_len))
          break;
      }
      if (on_msg)
        on_msg(msg);
    }
  } catch (...) {
    // 协议错误：静默关闭（对端不可信）
  }
}

void Connection::write_loop() {
  auto self = shared_from_this();
  while (true) {
    std::string frame;
    {
      std::unique_lock<std::mutex> lk(wmtx_);
      wcv_.wait(lk, [&] { return closing_.load() || !wqueue_.empty(); });
      if (closing_.load() && wqueue_.empty())
        return;
      frame = std::move(wqueue_.front());
      wqueue_.pop();
    }
    size_t sent = 0;
    while (sent < frame.size()) {
#ifdef _WIN32
      int r = ::send(static_cast<SOCKET>(fd_), frame.data() + sent,
                     static_cast<int>(frame.size() - sent), 0);
#else
      ssize_t r = ::send(fd_, frame.data() + sent, frame.size() - sent,
                         MSG_NOSIGNAL);
#endif
      if (r <= 0) {
        close();
        return;
      }
      sent += static_cast<size_t>(r);
    }
  }
}

bool Connection::send(const Message& msg) {
  if (!open_.load())
    return false;
  try {
    std::string frame = encode_frame(msg);
    std::lock_guard<std::mutex> lk(wmtx_);
    if (wqueue_.size() >= WRITE_QUEUE_HIGH_WATER)
      return false; // 背压：调用方应降速或稍后重试
    wqueue_.push(std::move(frame));
    wcv_.notify_one();
    return true;
  } catch (...) {
    return false;
  }
}

void Connection::close() {
  closing_ = true;
  open_ = false;
  bool fd_open = false;
  if (fd_closed_.compare_exchange_strong(fd_open, true)) {
#ifdef _WIN32
    shutdown(static_cast<SOCKET>(fd_), SD_BOTH);
    sock::close(fd_);
#else
    ::shutdown(fd_, SHUT_RDWR);
    sock::close(fd_);
#endif
  }
  wcv_.notify_all();
  // 若在自身读/写线程内被调用（如连接回调触发的析构），detach 自身避免
  // joinable 线程随对象销毁触发 std::terminate
  if (reader_.joinable()) {
    if (reader_.get_id() == std::this_thread::get_id())
      reader_.detach();
    else
      reader_.join();
  }
  if (writer_.joinable()) {
    if (writer_.get_id() == std::this_thread::get_id())
      writer_.detach();
    else
      writer_.join();
  }
}

// ---------- TcpServer ----------

TcpServer::~TcpServer() { stop(); }

bool TcpServer::listen(uint16_t port, ConnectionHandler on_conn) {
  sock::startup();
  on_conn_ = std::move(on_conn);
#ifdef _WIN32
  listen_fd_ = static_cast<sock::socket_t>(
      ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
#else
  listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
#endif
  if (listen_fd_ == sock::INVALID)
    return false;
  sock::set_reuseaddr(listen_fd_);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (::bind(sock::native(listen_fd_), reinterpret_cast<sockaddr*>(&addr),
             sizeof(addr)) != 0) {
    sock::close(listen_fd_);
    listen_fd_ = sock::INVALID;
    return false;
  }
  if (::listen(sock::native(listen_fd_), 64) != 0) {
    sock::close(listen_fd_);
    listen_fd_ = sock::INVALID;
    return false;
  }
  port_ = port;
  running_ = true;
  accept_thread_ = std::thread([this] { accept_loop(); });
  return true;
}

void TcpServer::accept_loop() {
  while (running_.load()) {
    sockaddr_in peer_addr{};
#ifdef _WIN32
    int len = sizeof(peer_addr);
    SOCKET c = ::accept(static_cast<SOCKET>(listen_fd_),
                        reinterpret_cast<sockaddr*>(&peer_addr), &len);
    if (c == INVALID_SOCKET)
      break;
    sock::socket_t fd = static_cast<sock::socket_t>(c);
#else
    socklen_t len = sizeof(peer_addr);
    int c = ::accept(listen_fd_, reinterpret_cast<sockaddr*>(&peer_addr), &len);
    if (c < 0)
      break;
    sock::socket_t fd = c;
#endif
    char ip[INET_ADDRSTRLEN] = {0};
    inet_ntop(AF_INET, &peer_addr.sin_addr, ip, sizeof(ip));
    std::string peer = std::string(ip) + ":" +
                       std::to_string(ntohs(peer_addr.sin_port));
    auto conn = std::make_shared<Connection>(fd, std::move(peer));
    if (on_conn_)
      on_conn_(conn);
  }
}

void TcpServer::stop() {
  running_ = false;
  if (listen_fd_ != sock::INVALID) {
#ifdef _WIN32
    shutdown(static_cast<SOCKET>(listen_fd_), SD_BOTH);
#else
    ::shutdown(listen_fd_, SHUT_RDWR);
#endif
    sock::close(listen_fd_);
    listen_fd_ = sock::INVALID;
  }
  if (accept_thread_.joinable())
    accept_thread_.join();
}

// ---------- tcp_connect ----------

std::shared_ptr<Connection> tcp_connect(const std::string& host, uint16_t port,
                                        int timeout_ms) {
  (void)timeout_ms; // 预留：当前为阻塞连接
  sock::startup();
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* res = nullptr;
  std::string port_str = std::to_string(port);
  if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0)
    return nullptr;

  sock::socket_t fd = sock::INVALID;
  for (addrinfo* p = res; p; p = p->ai_next) {
#ifdef _WIN32
    fd = static_cast<sock::socket_t>(
        ::socket(p->ai_family, p->ai_socktype, p->ai_protocol));
    if (fd == sock::INVALID)
      continue;
    if (::connect(static_cast<SOCKET>(fd), p->ai_addr,
                  static_cast<int>(p->ai_addrlen)) == 0)
      break;
    sock::close(fd);
    fd = sock::INVALID;
#else
    fd = ::socket(p->ai_family, p->ai_socktype, p->ai_protocol);
    if (fd == sock::INVALID)
      continue;
    if (::connect(fd, p->ai_addr, p->ai_addrlen) == 0)
      break;
    sock::close(fd);
    fd = sock::INVALID;
#endif
  }
  freeaddrinfo(res);
  if (fd == sock::INVALID)
    return nullptr;

  char ip[INET_ADDRSTRLEN] = {0};
  (void)ip;
  return std::make_shared<Connection>(fd, host + ":" + std::to_string(port));
}

} // namespace mahe
