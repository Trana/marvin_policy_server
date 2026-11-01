#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace marvin::native_io {
using Joints = std::array<double, 12>;

inline void finite(const Joints &values) {
  for (auto v : values) if (!std::isfinite(v))
    throw std::invalid_argument("Joint values must be finite");
}

struct OutputConfig {
  Joints defaults{}, lower{}, upper{};
  double scale = .25;
  bool smooth = false;

  void validate() const {
    finite(defaults); finite(lower); finite(upper);
    if (!std::isfinite(scale) || scale <= 0.)
      throw std::invalid_argument("Action scale must be finite and positive");
    for (size_t i = 0; i < defaults.size(); ++i) {
      if (lower[i] > defaults[i] || defaults[i] > upper[i] ||
          (smooth && (lower[i] == defaults[i] || upper[i] == defaults[i])))
        throw std::invalid_argument("Default position must be within joint limits");
    }
  }
};

class ActionOutput {
 public:
  explicit ActionOutput(OutputConfig config) : config_(config) { config_.validate(); }

  Joints map(const Joints &action, double factor) const {
    finite(action);
    if (!std::isfinite(factor) || factor < 0. || factor > 1.)
      throw std::invalid_argument("Ramp factor must be in [0,1]");
    Joints result{};
    for (size_t i = 0; i < action.size(); ++i) {
      // The policy runner validates/clips raw actions before ramping.
      const auto value = action[i] * factor;
      auto offset = value * config_.scale;
      if (config_.smooth) {
        const auto range = value >= 0. ? config_.upper[i] - config_.defaults[i] :
                                       config_.defaults[i] - config_.lower[i];
        offset = range * std::tanh(offset / range);
      }
      result[i] = std::clamp(config_.defaults[i] + offset, config_.lower[i], config_.upper[i]);
    }
    return result;
  }

  Joints lock(const Joints &positions) const {
    finite(positions);
    for (size_t i = 0; i < positions.size(); ++i)
      if (positions[i] < config_.lower[i] || positions[i] > config_.upper[i])
        throw std::invalid_argument("Locked output exceeds position limits");
    return positions;
  }

 private:
  OutputConfig config_;
};

// Mirrors ActivationManager's raw-action ramp semantics. ROS time is supplied by
// the owner. A clock rewind must reset the lifecycle and restart the ramp epoch.
class Ramp {
 public:
  Ramp(double activation_seconds, double deactivation_seconds)
      : up_(activation_seconds), down_(deactivation_seconds) {
    if (!std::isfinite(up_) || !std::isfinite(down_) || up_ < 0. || down_ < 0.)
      throw std::invalid_argument("Ramp durations must be finite and nonnegative");
  }
  void activate(double now) {
    check_time(now);
    if (!active_) { active_ = true; up_start_ = now; down_start_.reset(); }
  }
  void deactivate(double now, const Joints &raw_action) {
    check_time(now); finite(raw_action);
    if (!active_) return;
    active_ = false; up_start_.reset(); down_action_ = raw_action;
    down_start_ = down_ > 0. ? std::optional<double>(now) : std::nullopt;
  }
  void stop() { active_ = false; up_start_.reset(); down_start_.reset(); }
  bool active() const { return active_; }
  bool output_released() const { return !active_ && !down_start_; }
  double activation_factor(double now) {
    check_time(now);
    if (!active_) return 0.;
    if (!up_start_) return 1.;
    const auto dt = now - *up_start_;
    if (up_ <= 0. || dt >= up_) { up_start_.reset(); return 1.; }
    return std::clamp(dt / up_, 0., 1.);
  }
  std::optional<Joints> deactivation_action(double now) {
    check_time(now);
    if (active_ || !down_start_) return std::nullopt;
    const auto dt = now - *down_start_;
    if (dt >= down_) { down_start_.reset(); return std::nullopt; }
    auto result = down_action_;
    const auto factor = 1. - std::clamp(dt / down_, 0., 1.);
    for (auto &value : result) value *= factor;
    return result;
  }
 private:
  static void check_time(double now) {
    if (!std::isfinite(now) || now < 0.) throw std::invalid_argument("Invalid ramp time");
  }
  double up_, down_;
  bool active_ = false;
  std::optional<double> up_start_, down_start_;
  Joints down_action_{};
};
}  // namespace marvin::native_io
