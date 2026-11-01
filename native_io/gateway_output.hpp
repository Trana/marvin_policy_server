#pragma once
#include "action_output.hpp"
#include "protocol.hpp"
#include <optional>

namespace marvin::native_io {
// Supervisor identity is opaque. The numeric epoch belongs only to native IPC.
class GatewayOutput {
 public:
  bool prepare(const Packet &p, uint64_t now, bool released) {
    if (!released || p.token.empty() || p.token.size() > max_token_bytes ||
        p.values.size() != 12 || p.requested_ns <= now ||
        p.requested_ns - now > 5000000000ULL) return false;
    for (auto value : p.values) if (!std::isfinite(value)) return false;
    token_ = p.token;
    std::copy_n(p.values.begin(), 12, initial_.begin());
    expires_ = p.requested_ns; epoch_.reset(); sequence_ = 0;
    return true;
  }
  bool ready(uint64_t now) const { return !token_.empty() && !epoch_ && now <= expires_; }
  void start(uint64_t epoch, uint64_t ros) { epoch_ = epoch; started_ = ros; }
  void rebind(uint64_t epoch) { if (epoch_) epoch_ = epoch; }
  void stop() { epoch_.reset(); token_.clear(); }
  bool released() const { return !epoch_; }
  bool owns(uint64_t epoch) const { return epoch_ && *epoch_ == epoch; }
  Joints command(const Joints &targets, uint64_t epoch, uint64_t ros) {
    if (!owns(epoch) || ros < started_) throw std::logic_error("Gateway output is not active");
    const auto blend = std::min(1., (ros - started_) / 2e9);
    Joints result{};
    for (size_t i = 0; i < result.size(); ++i) result[i] = initial_[i] + blend * (targets[i] - initial_[i]);
    ++sequence_;
    return result;
  }
  const std::string &token() const { return token_; }
  uint64_t sequence() const { return sequence_; }
 private:
  std::string token_;
  Joints initial_{};
  uint64_t expires_ = 0, started_ = 0, sequence_ = 0;
  std::optional<uint64_t> epoch_;
};
}
