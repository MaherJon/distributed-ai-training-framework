#include "mahe/scheduler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace mahe {

static uint64_t now_ms() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

Scheduler::Scheduler() = default;
Scheduler::~Scheduler() { stop_watchdog(); }

uint32_t Scheduler::add_node(const std::string& addr, const std::string& name,
                             uint32_t cores, uint64_t mem_mb, bool has_avx2,
                             double score) {
  NodeInfo info;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    info.id = next_node_id_++;
    info.addr = addr;
    info.name = name;
    info.cores = cores;
    info.mem_mb = mem_mb;
    info.has_avx2 = has_avx2;
    info.score = score > 0 ? score : 1.0;
    info.alive = true;
    info.last_beat_ms = now_ms();
    nodes_[info.id] = info;
  }
  if (on_node_join_)
    on_node_join_(info);
  return info.id;
}

void Scheduler::remove_node(uint32_t node_id) {
  std::lock_guard<std::mutex> lk(mtx_);
  nodes_.erase(node_id);
  // 该节点未完成任务回收到 PENDING
  for (auto& [tid, t] : tasks_) {
    if (t.assigned_node == node_id && t.state == TaskState::ASSIGNED) {
      t.state = TaskState::PENDING;
      t.assigned_node = 0;
      t.progress = 0.0;
    }
  }
}

bool Scheduler::heartbeat(uint32_t node_id, double progress) {
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = nodes_.find(node_id);
  if (it == nodes_.end())
    return false;
  it->second.last_beat_ms = now_ms();
  it->second.alive = true;
  // 心跳顺带更新该节点名下任务进度
  for (auto& [tid, t] : tasks_) {
    if (t.assigned_node == node_id && t.state == TaskState::ASSIGNED)
      t.progress = progress;
  }
  return true;
}

std::vector<NodeInfo> Scheduler::nodes() const {
  std::lock_guard<std::mutex> lk(mtx_);
  std::vector<NodeInfo> out;
  out.reserve(nodes_.size());
  for (auto& [id, n] : nodes_)
    out.push_back(n);
  return out;
}

size_t Scheduler::alive_count() const {
  std::lock_guard<std::mutex> lk(mtx_);
  size_t n = 0;
  for (auto& [id, info] : nodes_)
    if (info.alive)
      ++n;
  return n;
}

std::map<uint32_t, uint64_t> Scheduler::weighted_split(uint64_t total) const {
  std::lock_guard<std::mutex> lk(mtx_);
  std::map<uint32_t, uint64_t> out;
  double sum = 0.0;
  for (auto& [id, n] : nodes_)
    if (n.alive)
      sum += n.score;
  if (sum <= 0.0 || total == 0)
    return out;

  // 先按比例下取整，再把余数按小数部分大小补齐（最大余数法）
  struct Rem {
    uint32_t id;
    uint64_t base;
    double frac;
  };
  std::vector<Rem> rems;
  uint64_t allocated = 0;
  for (auto& [id, n] : nodes_) {
    if (!n.alive)
      continue;
    double exact = static_cast<double>(total) * n.score / sum;
    uint64_t base = static_cast<uint64_t>(std::floor(exact));
    rems.push_back({id, base, exact - static_cast<double>(base)});
    allocated += base;
  }
  // 按小数部分降序分配剩余
  std::sort(rems.begin(), rems.end(),
            [](const Rem& a, const Rem& b) { return a.frac > b.frac; });
  uint64_t rest = total - allocated;
  for (size_t i = 0; i < rems.size() && rest > 0; ++i, --rest)
    ++rems[i].base;
  for (auto& r : rems)
    if (r.base > 0)
      out[r.id] = r.base;
  return out;
}

uint32_t Scheduler::create_task() {
  std::lock_guard<std::mutex> lk(mtx_);
  uint32_t id = next_task_id_++;
  tasks_[id] = Task{id, 0, TaskState::PENDING, 0.0};
  return id;
}

bool Scheduler::assign_task(uint32_t task_id, uint32_t node_id) {
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = tasks_.find(task_id);
  if (it == tasks_.end())
    return false;
  it->second.assigned_node = node_id;
  it->second.state = TaskState::ASSIGNED;
  it->second.progress = 0.0;
  return true;
}

bool Scheduler::task_progress(uint32_t task_id, double progress) {
  std::lock_guard<std::mutex> lk(mtx_);
  auto it = tasks_.find(task_id);
  if (it == tasks_.end())
    return false;
  it->second.progress = progress;
  return true;
}

bool Scheduler::task_done(uint32_t task_id) {
  Task t;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = tasks_.find(task_id);
    if (it == tasks_.end())
      return false;
    it->second.state = TaskState::DONE;
    it->second.progress = 1.0;
    t = it->second;
  }
  if (on_task_event_)
    on_task_event_(t);
  return true;
}

bool Scheduler::task_failed(uint32_t task_id) {
  Task t;
  {
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = tasks_.find(task_id);
    if (it == tasks_.end())
      return false;
    it->second.state = TaskState::PENDING; // 回收重派
    it->second.assigned_node = 0;
    it->second.progress = 0.0;
    t = it->second;
  }
  if (on_task_event_)
    on_task_event_(t);
  return true;
}

std::vector<Task> Scheduler::tasks() const {
  std::lock_guard<std::mutex> lk(mtx_);
  std::vector<Task> out;
  out.reserve(tasks_.size());
  for (auto& [id, t] : tasks_)
    out.push_back(t);
  return out;
}

void Scheduler::start_watchdog(uint64_t beat_timeout_ms,
                               NodeEventHandler on_node_dead) {
  stop_watchdog();
  on_node_dead_ = std::move(on_node_dead);
  watchdog_running_ = true;
  watchdog_ = std::thread([this, beat_timeout_ms] {
    while (watchdog_running_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      std::vector<NodeInfo> dead;
      {
        std::lock_guard<std::mutex> lk(mtx_);
        uint64_t now = now_ms();
        for (auto& [id, n] : nodes_) {
          if (n.alive && now - n.last_beat_ms > beat_timeout_ms) {
            n.alive = false;
            dead.push_back(n);
            // 回收其任务
            for (auto& [tid, t] : tasks_) {
              if (t.assigned_node == id && t.state == TaskState::ASSIGNED) {
                t.state = TaskState::PENDING;
                t.assigned_node = 0;
                t.progress = 0.0;
              }
            }
          }
        }
      }
      for (auto& n : dead)
        if (on_node_dead_)
          on_node_dead_(n);
    }
  });
}

void Scheduler::stop_watchdog() {
  watchdog_running_ = false;
  if (watchdog_.joinable())
    watchdog_.join();
}

} // namespace mahe
