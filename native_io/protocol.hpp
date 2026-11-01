#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>

// v2 targets the same little-endian Linux host on both sides of AF_UNIX IPC.
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Native policy protocol v2 requires a little-endian host"
#endif

namespace marvin::native_io {
constexpr uint32_t magic = 0x4f49504d;
constexpr uint32_t version = 2;
constexpr size_t header_size = 84;
constexpr size_t max_token_bytes = 256;
constexpr size_t max_values = 96;
constexpr size_t max_packet = header_size + max_values * sizeof(double) + max_token_bytes;
static_assert(sizeof(double) == 8);

enum class Kind : uint32_t {
  configure = 1, snapshot, result, activate, deactivate,
  invalidate, ack, fault, status, shutdown, commit, prepare
};

enum Flags : uint64_t {
  permitted = 1, active = 2, sensors_valid = 4, paused = 8, faulted = 16,
  output_valid = 32, encoder_timing = 64, success = 128,
  output_locked = 256, direct_output = 512, output_released = 1024
};

struct Packet {
  Kind kind = Kind::status;
  uint64_t sequence = 0, generation = 0, requested_ns = 0, ros_ns = 0;
  uint64_t joint_ns = 0, imu_ns = 0, flags = 0, trace_id = 0;
  std::vector<double> values;
  std::string token = {};
};

inline bool valid_kind(uint32_t value) { return value >= 1 && value <= 12; }

inline std::vector<uint8_t> encode(const Packet &p) {
  if (!valid_kind(static_cast<uint32_t>(p.kind)) || p.values.size() > max_values ||
      p.token.size() > max_token_bytes || (!p.token.empty() && p.kind != Kind::prepare))
    throw std::invalid_argument("Invalid packet kind or payload size");
  for (auto v : p.values)
    if (!std::isfinite(v)) throw std::invalid_argument("Non-finite packet payload");
  std::array<uint32_t, 5> prefix{magic, version, static_cast<uint32_t>(p.kind),
                               static_cast<uint32_t>(p.values.size()), static_cast<uint32_t>(p.token.size())};
  std::array<uint64_t, 8> fields{p.sequence, p.generation, p.requested_ns, p.ros_ns,
                               p.joint_ns, p.imu_ns, p.flags, p.trace_id};
  std::vector<uint8_t> bytes(header_size + p.values.size() * sizeof(double) + p.token.size());
  std::memcpy(bytes.data(), prefix.data(), sizeof(prefix));
  std::memcpy(bytes.data() + sizeof(prefix), fields.data(), sizeof(fields));
  if (!p.values.empty())
    std::memcpy(bytes.data() + header_size, p.values.data(), p.values.size() * sizeof(double));
  if (!p.token.empty())
    std::memcpy(bytes.data() + header_size + p.values.size() * sizeof(double), p.token.data(), p.token.size());
  return bytes;
}

inline Packet decode(const uint8_t *data, size_t size) {
  if (size < header_size || size > max_packet)
    throw std::invalid_argument("Invalid packet size");
  std::array<uint32_t, 5> prefix{};
  std::array<uint64_t, 8> fields{};
  std::memcpy(prefix.data(), data, sizeof(prefix));
  std::memcpy(fields.data(), data + sizeof(prefix), sizeof(fields));
  if (prefix[0] != magic || prefix[1] != version || !valid_kind(prefix[2]) ||
      prefix[3] > max_values || prefix[4] > max_token_bytes ||
      (prefix[4] && prefix[2] != static_cast<uint32_t>(Kind::prepare)) ||
      size != header_size + prefix[3] * sizeof(double) + prefix[4])
    throw std::invalid_argument("Invalid packet header");
  Packet p{static_cast<Kind>(prefix[2]), fields[0], fields[1], fields[2], fields[3],
           fields[4], fields[5], fields[6], fields[7], {}};
  p.values.resize(prefix[3]);
  if (!p.values.empty()) std::memcpy(p.values.data(), data + header_size, p.values.size() * sizeof(double));
  for (auto v : p.values)
    if (!std::isfinite(v)) throw std::invalid_argument("Non-finite packet payload");
  p.token.assign(reinterpret_cast<const char *>(data + header_size + prefix[3] * sizeof(double)), prefix[4]);
  return p;
}
}  // namespace marvin::native_io
