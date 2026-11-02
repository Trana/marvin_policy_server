#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace marvin::native_io {

struct Request {
  uint64_t sequence, generation, steady_ns, ros_ns;
};

enum class Fault { none, inference_expired, invalid_result, transport_closed };

struct ExpiryDiagnostic {
  const char *branch;
  uint64_t steady_ns, ros_ns, max_age_ns, generation;
  std::optional<Request> pending, accepted;
};

// This gate owns acceptance, not motors. False movement permission invalidates
// an old velocity-conditioned result, but still permits fresh balance inference.
// The worker must receive permission=false with the new request and zero velocity.
class Lifecycle {
 public:
  explicit Lifecycle(uint64_t period_ns = 20000000, uint64_t max_age_ns = 100000000)
      : period_(period_ns), max_age_(max_age_ns) {
    if (!period_ || !max_age_) throw std::invalid_argument("Timing must be positive");
  }

  void activate() {
    expiry_.reset();
    invalidate(); enabled_ = true; fault_ = Fault::none;
  }

  void stop() {
    expiry_.reset();
    invalidate(); enabled_ = false; fault_ = Fault::none;
  }

  void fail(Fault reason) {
    invalidate(); enabled_ = false; fault_ = reason;
  }

  void invalidate() {
    ++generation_; pending_.reset(); accepted_.reset(); deadline_.reset();
  }

  void observe(uint64_t steady_ns, uint64_t ros_ns, bool sensors_valid,
               bool movement_permitted) {
    // A steady-clock reversal violates this process's timing assumptions.
    if (last_steady_ && steady_ns < *last_steady_) {
      fail(Fault::invalid_result); return;
    }
    const bool was_paused = paused_;
    const bool rewind = last_ros_ && ros_ns < *last_ros_;
    clock_progress_ = !last_ros_ || ros_ns != *last_ros_;
    if (clock_progress_) {
      last_progress_ = steady_ns;
      paused_ = false;
    } else if (last_progress_ && steady_ns - *last_progress_ >= max_age_) {
      paused_ = true;
    }
    if (rewind || (paused_ != was_paused) ||
        (sensors_valid_ && !sensors_valid) ||
        (permission_known_ && movement_permitted != permission_)) invalidate();
    last_ros_ = ros_ns; last_steady_ = steady_ns;
    sensors_valid_ = sensors_valid;
    permission_ = movement_permitted; permission_known_ = true;
  }

  void enforce_deadlines(uint64_t steady_ns, uint64_t ros_ns) {
    if (!enabled_ || !sensors_valid_ || paused_ || !clock_progress_) return;
    // The pending request owns inference progress. A stale accepted result is
    // already suppressed by can_publish(); it must not fault a newer request
    // that still has time remaining in its own request-age budget.
    if (pending_) {
      if (!fresh(*pending_, steady_ns, ros_ns))
        expire("pending", steady_ns, ros_ns);
      return;
    }
    if (accepted_ && !fresh(*accepted_, steady_ns, ros_ns))
      expire("accepted", steady_ns, ros_ns);
  }

  std::optional<Request> request(uint64_t steady_ns, uint64_t ros_ns) {
    if (!enabled_ || !sensors_valid_ || paused_ || !clock_progress_ || pending_ ||
        (deadline_ && ros_ns < *deadline_)) return std::nullopt;
    pending_ = Request{++sequence_, generation_, steady_ns, ros_ns};
    // One phase-aligned deadline, never a catch-up burst.
    auto origin = deadline_.value_or(ros_ns);
    deadline_ = origin + ((ros_ns - origin) / period_ + 1) * period_;
    return pending_;
  }

  bool accept(const Request &reply, uint64_t steady_ns, uint64_t ros_ns) {
    if (!enabled_ || !sensors_valid_ || paused_ || !pending_ ||
        reply.sequence != pending_->sequence || reply.generation != generation_ ||
        reply.steady_ns != pending_->steady_ns || reply.ros_ns != pending_->ros_ns)
      return false;
    if (!fresh(reply, steady_ns, ros_ns)) { expire("arriving_result", steady_ns, ros_ns); return false; }
    accepted_ = pending_; pending_.reset();
    return true;
  }

  bool can_publish(uint64_t steady_ns, uint64_t ros_ns) const {
    return enabled_ && sensors_valid_ && !paused_ && clock_progress_ && accepted_ &&
           fresh(*accepted_, steady_ns, ros_ns);
  }

  bool enabled() const { return enabled_; }
  bool paused() const { return paused_; }
  bool movement_permitted() const { return permission_; }
  uint64_t generation() const { return generation_; }
  Fault fault() const { return fault_; }
  const std::optional<ExpiryDiagnostic> &expiry() const { return expiry_; }

 private:
  void expire(const char *branch, uint64_t steady_ns, uint64_t ros_ns) {
    // Preserve evidence before fail() invalidates request ownership.
    expiry_ = ExpiryDiagnostic{branch, steady_ns, ros_ns, max_age_, generation_, pending_, accepted_};
    fail(Fault::inference_expired);
  }
  bool fresh(const Request &r, uint64_t steady_ns, uint64_t ros_ns) const {
    return steady_ns >= r.steady_ns && steady_ns - r.steady_ns <= max_age_ &&
           ros_ns >= r.ros_ns && ros_ns - r.ros_ns <= max_age_;
  }
  uint64_t period_, max_age_, generation_ = 0, sequence_ = 0;
  bool enabled_ = false, sensors_valid_ = false, permission_ = false;
  bool permission_known_ = false, paused_ = false, clock_progress_ = false;
  Fault fault_ = Fault::none;
  std::optional<uint64_t> last_ros_, last_steady_, last_progress_, deadline_;
  std::optional<Request> pending_, accepted_;
  std::optional<ExpiryDiagnostic> expiry_;
};
}  // namespace marvin::native_io
