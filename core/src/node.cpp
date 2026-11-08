#include "mahe/node.hpp"

#include <chrono>
#include <condition_variable>
#include <sstream>
#include <thread>

namespace mahe {

// ---- 简单键值负载编解码（\n 分隔，k=v） ----
static std::string kv_encode(
    const std::vector<std::pair<std::string, std::string>>& kvs) {
  std::string out;
  for (auto& [k, v] : kvs) {
    out += k;
    out += '=';
    out += v;
    out += '\n';
  }
  return out;
}

static std::unordered_map<std::string, std::string>
kv_decode(const std::string& s) {
  std::unordered_map<std::string, std::string> out;
  size_t pos = 0;
  while (pos < s.size()) {
    size_t nl = s.find('\n', pos);
    if (nl == std::string::npos)
      nl = s.size();
    size_t eq = s.find('=', pos);
    if (eq != std::string::npos && eq < nl)
      out[s.substr(pos, eq - pos)] = s.substr(eq + 1, nl - eq - 1);
    pos = nl + 1;
  }
  return out;
}

static uint32_t detect_cores() {
  unsigned n = std::thread::hardware_concurrency();
  return n ? n : 1;
}

CoreNode::CoreNode() = default;
CoreNode::~CoreNode() { stop(); }

// ---------------- Master ----------------

bool CoreNode::start_master(const MasterConfig& cfg) {
  token_ = cfg.token;
  is_master_ = true;
  heartbeat_ms_ = 3000;

  scheduler_.set_on_node_join([](const NodeInfo&) {});
  if (!server_.listen(cfg.tcp_port, [this](std::shared_ptr<Connection> conn) {
        handle_master_connection(std::move(conn));
      })) {
    return false;
  }

  if (cfg.enable_beacon) {
    beacon_.start(cfg.cluster_name, cfg.tcp_port, token_fingerprint(cfg.token),
                  cfg.udp_port);
  }

  scheduler_.start_watchdog(cfg.heartbeat_timeout_ms,
                            [this](const NodeInfo& n) {
                              std::lock_guard<std::mutex> lk(conns_mtx_);
                              auto it = conns_.find(n.id);
                              if (it != conns_.end()) {
                                conn_to_id_.erase(it->second.get());
                                conns_.erase(it);
                              }
                              if (on_node_dead_user_)
                                on_node_dead_user_(n);
                            });

  running_ = true;
  return true;
}

void CoreNode::handle_master_connection(std::shared_ptr<Connection> conn) {
  Connection* raw = conn.get();
  conn->start(
      [this, conn](const Message& msg) { handle_master_message(conn, msg); },
      [this, raw]() {
        // 连接关闭：若已认证则摘除
        std::lock_guard<std::mutex> lk(conns_mtx_);
        auto it = conn_to_id_.find(raw);
        if (it != conn_to_id_.end()) {
          conns_.erase(it->second);
          conn_to_id_.erase(it);
          scheduler_.remove_node(it->second);
        }
      });
}

void CoreNode::handle_master_message(std::shared_ptr<Connection> conn,
                                     const Message& msg) {
  if (msg.type == MsgType::HELLO) {
    auto kv = kv_decode(msg.payload);
    // 令牌校验
    if (!token_.empty() && kv["token"] != token_) {
      Message bye{MsgType::BYE, 0, msg.request_id, "reason=bad_token\n"};
      conn->send(bye);
      conn->close();
      return;
    }
    NodeInfo info_snapshot;
    uint32_t id = scheduler_.add_node(
        conn->peer(), kv["name"].empty() ? conn->peer() : kv["name"],
        static_cast<uint32_t>(std::stoul(kv["cores"].empty() ? "1" : kv["cores"])),
        std::stoull(kv["mem_mb"].empty() ? "0" : kv["mem_mb"]),
        kv["avx2"] == "1",
        std::stod(kv["score"].empty() ? "1.0" : kv["score"]));
    {
      std::lock_guard<std::mutex> lk(conns_mtx_);
      conns_[id] = conn;
      conn_to_id_[conn.get()] = id;
    }
    Message ack{MsgType::HELLO_ACK, 0, msg.request_id,
                kv_encode({{"node_id", std::to_string(id)},
                           {"heartbeat_ms", std::to_string(heartbeat_ms_)}})};
    conn->send(ack);
    return;
  }

  // 未认证连接的消息一律忽略
  uint32_t node_id = 0;
  {
    std::lock_guard<std::mutex> lk(conns_mtx_);
    auto it = conn_to_id_.find(conn.get());
    if (it == conn_to_id_.end())
      return;
    node_id = it->second;
  }

  if (msg.type == MsgType::HEARTBEAT) {
    auto kv = kv_decode(msg.payload);
    double progress = kv.count("progress") ? std::stod(kv["progress"]) : 0.0;
    scheduler_.heartbeat(node_id, progress);
    return;
  }

  if (on_message_)
    on_message_(node_id, msg);
}

bool CoreNode::send_to(uint32_t node_id, const Message& msg) {
  std::lock_guard<std::mutex> lk(conns_mtx_);
  auto it = conns_.find(node_id);
  if (it == conns_.end() || !it->second->is_open())
    return false;
  return it->second->send(msg);
}

void CoreNode::broadcast(const Message& msg) {
  std::lock_guard<std::mutex> lk(conns_mtx_);
  for (auto& [id, conn] : conns_)
    conn->send(msg);
}

// ---------------- Worker ----------------

bool CoreNode::start_worker(const WorkerConfig& cfg) {
  is_master_ = false;
  token_ = cfg.token;
  heartbeat_ms_ = cfg.heartbeat_ms;

  std::string host = cfg.master_host;
  uint16_t port = cfg.master_port;

  // 自动发现：监听 beacon
  if (host.empty()) {
    std::string found_host;
    uint16_t found_port = 0;
    {
      std::mutex m;
      std::condition_variable cv;
      bool done = false;
      BeaconListener listener;
      std::string want_fp = token_fingerprint(cfg.token);
      bool started = listener.start(
          cfg.udp_port, [&](const BeaconInfo& info) {
            // 空 token 集群接受任意 beacon；否则比对指纹
            if (!cfg.token.empty() &&
                info.token_fingerprint != want_fp)
              return;
            std::lock_guard<std::mutex> lk(m);
            if (done)
              return;
            found_host = info.from_addr;
            found_port = info.tcp_port;
            done = true;
            cv.notify_one();
          });
      if (!started)
        return false;
      std::unique_lock<std::mutex> lk(m);
      if (!cv.wait_for(lk,
                       std::chrono::milliseconds(cfg.discovery_timeout_ms),
                       [&] { return done; })) {
        return false; // 发现超时
      }
    }
    host = found_host;
    port = found_port;
  }

  master_conn_ = tcp_connect(host, port);
  if (!master_conn_)
    return false;

  master_conn_->start(
      [this](const Message& msg) { handle_worker_message(msg); },
      [this] { running_ = false; });

  // HELLO 握手
  uint32_t cores = cfg.cores ? cfg.cores : detect_cores();
  Message hello{MsgType::HELLO, 0, 1,
                kv_encode({{"token", cfg.token},
                           {"name", cfg.node_name},
                           {"cores", std::to_string(cores)},
                           {"mem_mb", std::to_string(cfg.mem_mb)},
                           {"avx2", "1"}, // 由 kernel 模块在绑定层修正
                           {"score", std::to_string(cfg.score)}})};
  if (!master_conn_->send(hello)) {
    master_conn_->close();
    master_conn_.reset();
    return false;
  }

  running_ = true;
  hb_thread_ = std::thread([this] { heartbeat_loop(); });
  return true;
}

void CoreNode::handle_worker_message(const Message& msg) {
  if (msg.type == MsgType::HELLO_ACK) {
    auto kv = kv_decode(msg.payload);
    self_id_ = static_cast<uint32_t>(
        std::stoul(kv["node_id"].empty() ? "0" : kv["node_id"]));
    if (kv.count("heartbeat_ms"))
      heartbeat_ms_ = std::stoull(kv["heartbeat_ms"]);
    return;
  }
  if (msg.type == MsgType::BYE) {
    stop();
    return;
  }
  if (on_message_)
    on_message_(0, msg);
}

void CoreNode::heartbeat_loop() {
  while (running_.load()) {
    Message hb{MsgType::HEARTBEAT, 0, 0,
               kv_encode({{"progress",
                           std::to_string(progress_.load())}})};
    if (master_conn_ && master_conn_->is_open()) {
      if (!master_conn_->send(hb))
        break;
    } else {
      break;
    }
    for (uint64_t i = 0; i < heartbeat_ms_ / 100 && running_.load(); ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  running_ = false;
}

bool CoreNode::send_to_master(const Message& msg) {
  if (!master_conn_ || !master_conn_->is_open())
    return false;
  return master_conn_->send(msg);
}

void CoreNode::report_progress(double p) { progress_ = p; }

// ---------------- 通用 ----------------

void CoreNode::stop() {
  bool was_running = running_.exchange(false);
  (void)was_running;
  if (master_conn_) {
    master_conn_->close();
    master_conn_.reset();
  }
  server_.stop();
  beacon_.stop();
  scheduler_.stop_watchdog();
  {
    // 先取出连接再关：close() 会 join 读线程，而读线程的关闭回调需要 conns_mtx_
    std::vector<std::shared_ptr<Connection>> to_close;
    {
      std::lock_guard<std::mutex> lk(conns_mtx_);
      for (auto& [id, conn] : conns_)
        to_close.push_back(conn);
      conns_.clear();
      conn_to_id_.clear();
    }
    for (auto& conn : to_close)
      conn->close();
  }
  if (hb_thread_.joinable() &&
      hb_thread_.get_id() != std::this_thread::get_id())
    hb_thread_.join();
}

void CoreNode::set_message_handler(MessageHandler cb) {
  on_message_ = std::move(cb);
}
void CoreNode::set_on_node_join(NodeEventHandler cb) {
  scheduler_.set_on_node_join(std::move(cb));
}
void CoreNode::set_on_node_dead(NodeEventHandler cb) {
  on_node_dead_user_ = std::move(cb);
}

} // namespace mahe
