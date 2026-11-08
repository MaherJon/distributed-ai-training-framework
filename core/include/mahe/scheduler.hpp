// MAHE 调度器（Master 侧）：节点注册、算力加权分片、心跳看门狗、任务跟踪
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mahe {

struct NodeInfo {
  uint32_t id{0};
  std::string addr;    // ip:port
  std::string name;
  uint32_t cores{0};
  uint64_t mem_mb{0};
  bool has_avx2{false};
  double score{1.0};   // 微基准算力分
  bool alive{true};
  uint64_t last_beat_ms{0};
};

enum class TaskState { PENDING, ASSIGNED, DONE, FAILED };

struct Task {
  uint32_t task_id{0};
  uint32_t assigned_node{0};
  TaskState state{TaskState::PENDING};
  double progress{0.0}; // 0..1
};

class Scheduler {
public:
  using NodeEventHandler = std::function<void(const NodeInfo&)>;
  using TaskEventHandler = std::function<void(const Task&)>;

  Scheduler();
  ~Scheduler();

  // ---- 节点管理 ----
  uint32_t add_node(const std::string& addr, const std::string& name,
                    uint32_t cores, uint64_t mem_mb, bool has_avx2,
                    double score);
  void remove_node(uint32_t node_id);
  bool heartbeat(uint32_t node_id, double progress);
  std::vector<NodeInfo> nodes() const;
  size_t alive_count() const;

  // ---- 加权分片：shard_i ∝ score_i / Σscores，总量 total ----
  // 返回 node_id -> 分片大小；仅分配给存活节点
  std::map<uint32_t, uint64_t> weighted_split(uint64_t total) const;

  // ---- 任务跟踪 ----
  uint32_t create_task();
  bool assign_task(uint32_t task_id, uint32_t node_id);
  bool task_progress(uint32_t task_id, double progress);
  bool task_done(uint32_t task_id);
  bool task_failed(uint32_t task_id); // 失败 → 回到 PENDING 等重派
  std::vector<Task> tasks() const;

  // ---- 看门狗：beat_timeout_ms 未心跳判死，触发 on_node_dead ----
  void start_watchdog(uint64_t beat_timeout_ms, NodeEventHandler on_node_dead);
  void stop_watchdog();

  void set_on_node_join(NodeEventHandler cb) { on_node_join_ = std::move(cb); }
  void set_on_task_event(TaskEventHandler cb) { on_task_event_ = std::move(cb); }

private:
  mutable std::mutex mtx_;
  uint32_t next_node_id_{1};
  uint32_t next_task_id_{1};
  std::map<uint32_t, NodeInfo> nodes_;
  std::map<uint32_t, Task> tasks_;

  std::thread watchdog_;
  std::atomic<bool> watchdog_running_{false};

  NodeEventHandler on_node_join_;
  NodeEventHandler on_node_dead_;
  TaskEventHandler on_task_event_;
};

} // namespace mahe
