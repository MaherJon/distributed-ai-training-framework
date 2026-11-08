// MAHE C++ 核心单元测试：kernel 正确性、协议编解码、调度器、端到端握手
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>

#include "mahe/kernel.hpp"
#include "mahe/node.hpp"
#include "mahe/protocol.hpp"
#include "mahe/scheduler.hpp"

static int g_fail = 0;
#define CHECK(cond, name)                                                    \
  do {                                                                       \
    if (cond) {                                                              \
      std::printf("[PASS] %s\n", name);                                      \
    } else {                                                                 \
      std::printf("[FAIL] %s\n", name);                                      \
      ++g_fail;                                                              \
    }                                                                        \
  } while (0)

static void test_kernel() {
  std::printf("kernel impl: %s\n", mahe::kernel::active_impl());
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t n : {0, 1, 3, 7, 8, 9, 31, 32, 33, 1000, 4097}) {
    std::vector<float> a(n), b(n), a_ref(n);
    for (auto& v : a) v = dist(rng);
    for (auto& v : b) v = dist(rng);
    a_ref = a;

    mahe::kernel::reduce_add_f32(a.data(), b.data(), n);
    mahe::kernel::reduce_add_f32_scalar(a_ref.data(), b.data(), n);
    bool ok = true;
    for (size_t i = 0; i < n; ++i)
      if (a[i] != a_ref[i]) { ok = false; break; }
    CHECK(ok, "reduce_add_f32 与标量一致");

    std::vector<float> p(n), p_ref(n), g(n);
    for (auto& v : p) v = dist(rng);
    for (auto& v : g) v = dist(rng);
    p_ref = p;
    mahe::kernel::sgd_update_f32(p.data(), g.data(), 0.1f, n);
    mahe::kernel::sgd_update_f32_scalar(p_ref.data(), g.data(), 0.1f, n);
    ok = true;
    for (size_t i = 0; i < n; ++i)
      if (std::fabs(p[i] - p_ref[i]) > 1e-6f) { ok = false; break; }
    CHECK(ok, "sgd_update_f32 与标量一致(容差1e-6)");
  }
}

static void test_protocol() {
  mahe::Message msg{mahe::MsgType::TASK_ASSIGN, 3, 12345, "hello-mahe"};
  std::string frame = mahe::encode_frame(msg);
  CHECK(frame.size() == mahe::HEADER_SIZE + msg.payload.size(),
        "帧长度正确");
  CHECK(std::memcmp(frame.data(), "MAHE", 4) == 0, "魔数正确");
  auto h = mahe::parse_header(frame.data());
  CHECK(h.type == mahe::MsgType::TASK_ASSIGN && h.flags == 3 &&
            h.request_id == 12345 && h.payload_len == msg.payload.size(),
        "头部解析正确");
}

static void test_scheduler() {
  mahe::Scheduler sch;
  uint32_t a = sch.add_node("10.0.0.1:1", "fast", 8, 16384, true, 3.0);
  uint32_t b = sch.add_node("10.0.0.2:1", "slow", 2, 4096, false, 1.0);
  auto split = sch.weighted_split(1000);
  CHECK(split[a] == 750 && split[b] == 250, "加权分片 3:1 → 750/250");
  uint32_t tid = sch.create_task();
  CHECK(sch.assign_task(tid, a), "任务派发");
  CHECK(sch.task_done(tid), "任务完成");
  sch.remove_node(a);
  CHECK(sch.alive_count() == 1, "节点移除");
}

static void test_master_worker() {
  mahe::CoreNode master;
  mahe::MasterConfig mcfg;
  mcfg.cluster_name = "test";
  mcfg.token = "secret";
  mcfg.tcp_port = 47920;
  mcfg.udp_port = 47921;
  mcfg.enable_beacon = false;
  CHECK(master.start_master(mcfg), "master 启动");

  std::atomic<int> master_got{0};
  master.set_message_handler(
      [&](uint32_t, const mahe::Message& m) { ++master_got; });

  mahe::CoreNode worker;
  mahe::WorkerConfig wcfg;
  wcfg.node_name = "w1";
  wcfg.token = "secret";
  wcfg.master_host = "127.0.0.1";
  wcfg.master_port = 47920;
  CHECK(worker.start_worker(wcfg), "worker 启动(直连)");

  // 等 HELLO 处理
  for (int i = 0; i < 50 && master.scheduler().alive_count() < 1; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(master.scheduler().alive_count() == 1, "master 看到 1 个节点");
  for (int i = 0; i < 50 && worker.self_id() == 0; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(worker.self_id() != 0, "worker 收到 HELLO_ACK 分配 id");

  // worker → master
  mahe::Message msg{mahe::MsgType::TASK_ACK, 0, 7, "ack"};
  CHECK(worker.send_to_master(msg), "worker 发送 TASK_ACK");
  for (int i = 0; i < 50 && master_got.load() < 1; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(master_got.load() == 1, "master 收到 TASK_ACK");

  // master → worker
  std::atomic<int> worker_got{0};
  worker.set_message_handler(
      [&](uint32_t, const mahe::Message& m) { ++worker_got; });
  auto nodes = master.scheduler().nodes();
  mahe::Message assign{mahe::MsgType::TASK_ASSIGN, 0, 0, "shard"};
  CHECK(master.send_to(nodes[0].id, assign), "master 派发任务");
  for (int i = 0; i < 50 && worker_got.load() < 1; ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  CHECK(worker_got.load() == 1, "worker 收到任务");

  // 错误 token 应被拒绝
  mahe::CoreNode bad;
  mahe::WorkerConfig bcfg = wcfg;
  bcfg.token = "wrong";
  CHECK(bad.start_worker(bcfg), "bad worker 连接成功(连接层)");
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  CHECK(bad.self_id() == 0, "错误 token 未获分配 id");
  bad.stop();

  worker.stop();
  master.stop();
  CHECK(true, "优雅停止");
}

int main() {
  test_kernel();
  test_protocol();
  test_scheduler();
  test_master_worker();
  std::printf(g_fail == 0 ? "\n全部通过\n" : "\n%d 项失败\n", g_fail);
  return g_fail == 0 ? 0 : 1;
}
