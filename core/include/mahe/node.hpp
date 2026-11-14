// MAHE 节点门面：整合 transport / discovery / scheduler
// Master：TCP 服务 + beacon 广播 + 调度器 + HELLO 认证
// Worker：发现/直连 Master + HELLO 握手 + 周期心跳
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#include "mahe/discovery.hpp"
#include "mahe/scheduler.hpp"
#include "mahe/transport.hpp"

namespace mahe {

// HELLO 负载（文本键值，\n 分隔）：token/name/cores/mem_mb/avx2/score
// HELLO_ACK 负载：node_id=<id>\nheartbeat_ms=<ms>

struct MasterConfig {
  std::string cluster_name = "mahe-cluster";
  std::string token;                 // 集群令牌，空 = 不校验（仅信任网络）
  uint16_t tcp_port = 47820;
  uint16_t udp_port = 47821;
  uint64_t heartbeat_timeout_ms = 10000;
  bool enable_beacon = true;
};

struct WorkerConfig {
  std::string node_name;
  std::string token;
  std::string master_host;           // 为空则先听 beacon 自动发现
  uint16_t master_port = 47820;
  uint16_t udp_port = 47821;         // 自动发现监听口
  uint32_t cores = 0;                // 0 = 自动检测
  uint64_t mem_mb = 0;
  double score = 1.0;                // 微基准分
  uint64_t heartbeat_ms = 3000;
  uint32_t discovery_timeout_ms = 10000;
};

class CoreNode {
public:
  // payload 消息回调：(node_id, msg)；Master 侧 node_id 为来源节点，Worker 侧恒 0
  using MessageHandler =
      std::function<void(uint32_t node_id, const Message& msg)>;
  using NodeEventHandler = std::function<void(const NodeInfo&)>;

  CoreNode();
  ~CoreNode();

  bool start_master(const MasterConfig& cfg);
  bool start_worker(const WorkerConfig& cfg);
  void stop();

  bool is_master() const { return is_master_; }
  bool is_running() const { return running_.load(); }
  uint32_t self_id() const { return self_id_; } // Worker：Master 分配的 id

  // Master API
  bool send_to(uint32_t node_id, const Message& msg);
  void broadcast(const Message& msg);
  Scheduler& scheduler() { return scheduler_; }

  // Worker API
  bool send_to_master(const Message& msg);
  void report_progress(double progress); // 内部随心跳上报

  void set_message_handler(MessageHandler cb);
  void set_on_node_join(NodeEventHandler cb);
  void set_on_node_dead(NodeEventHandler cb);

private:
  void handle_master_connection(std::shared_ptr<Connection> conn);
  void handle_master_message(std::shared_ptr<Connection> conn,
                             const Message& msg);
  void handle_worker_message(const Message& msg);
  void heartbeat_loop();

  std::atomic<bool> running_{false};
  bool is_master_{false};
  std::string token_;
  uint64_t heartbeat_ms_{3000};

  // Master
  TcpServer server_;
  BeaconBroadcaster beacon_;
  Scheduler scheduler_;
  std::mutex conns_mtx_;
  std::unordered_map<uint32_t, std::shared_ptr<Connection>> conns_; // id→连接
  std::unordered_map<Connection*, uint32_t> conn_to_id_;

  // Worker
  std::shared_ptr<Connection> master_conn_;
  std::thread hb_thread_;
  std::atomic<double> progress_{0.0};
  uint32_t self_id_{0};

  MessageHandler on_message_;
  NodeEventHandler on_node_dead_user_;
};

} // namespace mahe
