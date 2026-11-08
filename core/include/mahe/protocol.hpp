// MAHE 应用层帧协议（v1）
// 16B 固定头部，全小端：
//   [0..3]  魔数 "MAHE"
//   [4..5]  版本 u16 (=1)
//   [6]     消息类型 u8
//   [7]     flags u8 (bit0=压缩预留 bit1=分片续传 bit2=请求-响应)
//   [8..11] payload_len u32
//   [12..15] request_id u32（请求-响应配对，心跳/广播为 0）
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace mahe {

inline constexpr uint8_t PROTOCOL_VERSION = 1;
inline constexpr size_t HEADER_SIZE = 16;
inline constexpr uint32_t MAX_PAYLOAD = 64u * 1024u * 1024u; // 64MB 帧上限

enum class MsgType : uint8_t {
  HELLO = 0x01,
  HELLO_ACK = 0x02,
  HEARTBEAT = 0x03,
  TASK_ASSIGN = 0x10,
  TASK_PROGRESS = 0x11,
  GRAD_PUSH = 0x12,
  PARAM_SYNC = 0x13,
  TASK_ACK = 0x14,
  TASK_FAIL = 0x15,
  CHECKPOINT_NOTICE = 0x20,
  BYE = 0x7F,
};

struct Message {
  MsgType type{MsgType::HEARTBEAT};
  uint8_t flags{0};
  uint32_t request_id{0};
  std::string payload; // 二进制负载（可为空）
};

inline void write_u16le(std::string& out, uint16_t v) {
  out.push_back(static_cast<char>(v & 0xFF));
  out.push_back(static_cast<char>((v >> 8) & 0xFF));
}

inline void write_u32le(std::string& out, uint32_t v) {
  for (int i = 0; i < 4; ++i)
    out.push_back(static_cast<char>((v >> (8 * i)) & 0xFF));
}

inline uint16_t read_u16le(const char* p) {
  return static_cast<uint16_t>(static_cast<uint8_t>(p[0])) |
         (static_cast<uint16_t>(static_cast<uint8_t>(p[1])) << 8);
}

inline uint32_t read_u32le(const char* p) {
  uint32_t v = 0;
  for (int i = 0; i < 4; ++i)
    v |= static_cast<uint32_t>(static_cast<uint8_t>(p[i])) << (8 * i);
  return v;
}

// 序列化一帧（header + payload）
inline std::string encode_frame(const Message& msg) {
  if (msg.payload.size() > MAX_PAYLOAD)
    throw std::length_error("payload exceeds MAX_PAYLOAD");
  std::string out;
  out.reserve(HEADER_SIZE + msg.payload.size());
  out.append("MAHE", 4);
  write_u16le(out, PROTOCOL_VERSION);
  out.push_back(static_cast<char>(msg.type));
  out.push_back(static_cast<char>(msg.flags));
  write_u32le(out, static_cast<uint32_t>(msg.payload.size()));
  write_u32le(out, msg.request_id);
  out.append(msg.payload);
  return out;
}

// 解析 16B 头部；非法头部抛异常（调用方应关闭连接）
struct FrameHeader {
  MsgType type;
  uint8_t flags;
  uint32_t payload_len;
  uint32_t request_id;
};

inline FrameHeader parse_header(const char* buf) {
  if (std::memcmp(buf, "MAHE", 4) != 0)
    throw std::runtime_error("bad magic");
  if (read_u16le(buf + 4) != PROTOCOL_VERSION)
    throw std::runtime_error("unsupported protocol version");
  FrameHeader h;
  h.type = static_cast<MsgType>(static_cast<uint8_t>(buf[6]));
  h.flags = static_cast<uint8_t>(buf[7]);
  h.payload_len = read_u32le(buf + 8);
  h.request_id = read_u32le(buf + 12);
  if (h.payload_len > MAX_PAYLOAD)
    throw std::runtime_error("payload too large");
  return h;
}

// ---- 张量 blob 编码：[dtype:1B][ndim:1B][shape: ndim×u64le][raw data] ----
// dtype: 0=f32, 1=f64, 2=i32, 3=u8, 4=i8(量化)

inline std::string encode_tensor_f32(const std::vector<int64_t>& shape,
                                     const float* data, size_t count) {
  std::string out;
  out.reserve(2 + shape.size() * 8 + count * 4);
  out.push_back(0); // f32
  out.push_back(static_cast<char>(shape.size()));
  for (int64_t d : shape) {
    uint64_t u = static_cast<uint64_t>(d);
    for (int i = 0; i < 8; ++i)
      out.push_back(static_cast<char>((u >> (8 * i)) & 0xFF));
  }
  const char* raw = reinterpret_cast<const char*>(data);
  out.append(raw, count * sizeof(float));
  return out;
}

} // namespace mahe
