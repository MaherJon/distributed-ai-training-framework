// MAHE C++ 核心 Python 绑定（pybind11）
// 暴露：CoreNode（master/worker）、kernel（NumPy 零拷贝）、协议常量
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mahe/kernel.hpp"
#include "mahe/node.hpp"

namespace py = pybind11;

namespace {

// 将 Python 回调包成线程安全形式（C++ 线程触发时夺取 GIL）
template <typename F> class GilCallback {
public:
  explicit GilCallback(F f) : f_(std::move(f)) {}
  template <typename... Args> void operator()(Args&&... args) {
    py::gil_scoped_acquire gil;
    try {
      f_(std::forward<Args>(args)...);
    } catch (const py::error_already_set& e) {
      // 回调异常不应终止 C++ 读线程；打印后吞掉
      PyErr_Print();
    }
  }

private:
  F f_;
};

} // namespace

PYBIND11_MODULE(mahe_core, m) {
  m.doc() = "MAHE distributed training core (C++/asm)";

  // ---- 协议枚举 ----
  py::enum_<mahe::MsgType>(m, "MsgType")
      .value("HELLO", mahe::MsgType::HELLO)
      .value("HELLO_ACK", mahe::MsgType::HELLO_ACK)
      .value("HEARTBEAT", mahe::MsgType::HEARTBEAT)
      .value("TASK_ASSIGN", mahe::MsgType::TASK_ASSIGN)
      .value("TASK_PROGRESS", mahe::MsgType::TASK_PROGRESS)
      .value("GRAD_PUSH", mahe::MsgType::GRAD_PUSH)
      .value("PARAM_SYNC", mahe::MsgType::PARAM_SYNC)
      .value("TASK_ACK", mahe::MsgType::TASK_ACK)
      .value("TASK_FAIL", mahe::MsgType::TASK_FAIL)
      .value("CHECKPOINT_NOTICE", mahe::MsgType::CHECKPOINT_NOTICE)
      .value("BYE", mahe::MsgType::BYE);

  py::class_<mahe::Message>(m, "Message")
      .def(py::init<>())
      .def_readwrite("type", &mahe::Message::type)
      .def_readwrite("flags", &mahe::Message::flags)
      .def_readwrite("request_id", &mahe::Message::request_id)
      // payload 是二进制 → 必须以 bytes 交互（py::str 会做 UTF-8 校验）
      .def_property(
          "payload",
          [](const mahe::Message& msg) { return py::bytes(msg.payload); },
          [](mahe::Message& msg, const py::bytes& b) {
            msg.payload = std::string(b);
          });

  py::class_<mahe::NodeInfo>(m, "NodeInfo")
      .def_readonly("id", &mahe::NodeInfo::id)
      .def_readonly("addr", &mahe::NodeInfo::addr)
      .def_readonly("name", &mahe::NodeInfo::name)
      .def_readonly("cores", &mahe::NodeInfo::cores)
      .def_readonly("mem_mb", &mahe::NodeInfo::mem_mb)
      .def_readonly("has_avx2", &mahe::NodeInfo::has_avx2)
      .def_readonly("score", &mahe::NodeInfo::score)
      .def_readonly("alive", &mahe::NodeInfo::alive);

  py::class_<mahe::Task>(m, "Task")
      .def_readonly("task_id", &mahe::Task::task_id)
      .def_readonly("assigned_node", &mahe::Task::assigned_node)
      .def_readonly("state", &mahe::Task::state)
      .def_readonly("progress", &mahe::Task::progress);

  py::enum_<mahe::TaskState>(m, "TaskState")
      .value("PENDING", mahe::TaskState::PENDING)
      .value("ASSIGNED", mahe::TaskState::ASSIGNED)
      .value("DONE", mahe::TaskState::DONE)
      .value("FAILED", mahe::TaskState::FAILED);

  py::class_<mahe::MasterConfig>(m, "MasterConfig")
      .def(py::init<>())
      .def_readwrite("cluster_name", &mahe::MasterConfig::cluster_name)
      .def_readwrite("token", &mahe::MasterConfig::token)
      .def_readwrite("tcp_port", &mahe::MasterConfig::tcp_port)
      .def_readwrite("udp_port", &mahe::MasterConfig::udp_port)
      .def_readwrite("heartbeat_timeout_ms",
                     &mahe::MasterConfig::heartbeat_timeout_ms)
      .def_readwrite("enable_beacon", &mahe::MasterConfig::enable_beacon);

  py::class_<mahe::WorkerConfig>(m, "WorkerConfig")
      .def(py::init<>())
      .def_readwrite("node_name", &mahe::WorkerConfig::node_name)
      .def_readwrite("token", &mahe::WorkerConfig::token)
      .def_readwrite("master_host", &mahe::WorkerConfig::master_host)
      .def_readwrite("master_port", &mahe::WorkerConfig::master_port)
      .def_readwrite("udp_port", &mahe::WorkerConfig::udp_port)
      .def_readwrite("cores", &mahe::WorkerConfig::cores)
      .def_readwrite("mem_mb", &mahe::WorkerConfig::mem_mb)
      .def_readwrite("score", &mahe::WorkerConfig::score)
      .def_readwrite("heartbeat_ms", &mahe::WorkerConfig::heartbeat_ms)
      .def_readwrite("discovery_timeout_ms",
                     &mahe::WorkerConfig::discovery_timeout_ms);

  py::class_<mahe::Scheduler>(m, "Scheduler")
      .def("nodes", &mahe::Scheduler::nodes)
      .def("alive_count", &mahe::Scheduler::alive_count)
      // 节点移出分组：只摘除调度记录，连接仍可被 master 侧关闭
      .def("remove_node", &mahe::Scheduler::remove_node)
      .def("weighted_split", &mahe::Scheduler::weighted_split)
      .def("create_task", &mahe::Scheduler::create_task)
      .def("assign_task", &mahe::Scheduler::assign_task)
      .def("task_done", &mahe::Scheduler::task_done)
      .def("task_failed", &mahe::Scheduler::task_failed)
      .def("tasks", &mahe::Scheduler::tasks);

  py::class_<mahe::CoreNode>(m, "CoreNode")
      .def(py::init<>())
      .def("start_master", &mahe::CoreNode::start_master)
      .def("start_worker", &mahe::CoreNode::start_worker)
      .def("stop", &mahe::CoreNode::stop)
      .def("is_master", &mahe::CoreNode::is_master)
      // Worker 被 master 移出分组 / 连接断开后，用于让进程正常退出
      .def("is_running", &mahe::CoreNode::is_running)
      .def("self_id", &mahe::CoreNode::self_id)
      .def("send_to", &mahe::CoreNode::send_to)
      .def("broadcast", &mahe::CoreNode::broadcast)
      .def("send_to_master", &mahe::CoreNode::send_to_master)
      .def("report_progress", &mahe::CoreNode::report_progress)
      .def("scheduler", &mahe::CoreNode::scheduler,
           py::return_value_policy::reference_internal)
      .def("set_message_handler",
           [](mahe::CoreNode& self,
              std::function<void(uint32_t, const mahe::Message&)> cb) {
             self.set_message_handler(
                 GilCallback<std::function<void(uint32_t, const mahe::Message&)>>(
                     std::move(cb)));
           })
      .def("set_on_node_join",
           [](mahe::CoreNode& self,
              std::function<void(const mahe::NodeInfo&)> cb) {
             self.set_on_node_join(
                 GilCallback<std::function<void(const mahe::NodeInfo&)>>(
                     std::move(cb)));
           })
      .def("set_on_node_dead",
           [](mahe::CoreNode& self,
              std::function<void(const mahe::NodeInfo&)> cb) {
             self.set_on_node_dead(
                 GilCallback<std::function<void(const mahe::NodeInfo&)>>(
                     std::move(cb)));
           });

  // ---- kernel：NumPy 零拷贝 ----
  m.def("reduce_add_f32",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> dst,
           py::array_t<float, py::array::c_style | py::array::forcecast> src) {
          auto d = dst.mutable_unchecked<1>();
          auto s = src.unchecked<1>();
          if (d.shape(0) != s.shape(0))
            throw std::invalid_argument("shape mismatch");
          py::gil_scoped_release release;
          mahe::kernel::reduce_add_f32(dst.mutable_data(), src.data(),
                                       static_cast<size_t>(d.shape(0)));
        },
        "dst += src（AVX2 加速，原地）");

  m.def("sgd_update_f32",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> param,
           py::array_t<float, py::array::c_style | py::array::forcecast> grad,
           float lr) {
          auto p = param.mutable_unchecked<1>();
          auto g = grad.unchecked<1>();
          if (p.shape(0) != g.shape(0))
            throw std::invalid_argument("shape mismatch");
          py::gil_scoped_release release;
          mahe::kernel::sgd_update_f32(param.mutable_data(), grad.data(), lr,
                                       static_cast<size_t>(p.shape(0)));
        },
        "param -= lr * grad（AVX2 加速，原地）");

  m.def("kernel_impl", &mahe::kernel::active_impl,
        "当前内核实现：avx2 / scalar");

  m.def("token_fingerprint", &mahe::token_fingerprint);

  m.def("encode_tensor_f32",
        [](std::vector<int64_t> shape,
           py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
          auto r = arr.unchecked();
          std::string blob = mahe::encode_tensor_f32(
              shape, arr.data(), static_cast<size_t>(r.size()));
          return py::bytes(blob);
        },
        "张量 → blob 字节（[dtype][ndim][shape][raw]）");
}
