// Opt-in native sensor/command owner. No CAN interface or motor activation.
#include "action_output.hpp"
#include "lifecycle.hpp"
#include "protocol.hpp"
#include "profile.hpp"
#include "gateway_output.hpp"
#include <runtime_interfaces/msg/executor_command.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/create_timer.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <control_msgs/msg/dynamic_joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <marvin_trace_msgs/msg/traced_float64_multi_array.hpp>
#include <marvin_trace_msgs/msg/trace_event.hpp>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <set>
#include <sstream>

namespace mio = marvin::native_io;
using mio::Packet;
using mio::Kind;
using mio::Flags;

uint64_t steady_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

class NativeIo : public rclcpp::Node {
 public:
  explicit NativeIo(int fd) : Node("marvin_policy_io"), fd_(fd), boot_(steady_ns()),
      metric_start_(boot_) {
    names_ = declare_parameter<std::vector<std::string>>("joint_names", std::vector<std::string>{});
    if (names_.size() != 12 || std::set<std::string>(names_.begin(), names_.end()).size() != 12)
      throw std::invalid_argument("Native policy requires twelve unique joint names");
    auto joint_topic = declare_parameter<std::string>("joint_states_topic", "/policy_probe/joints");
    auto dynamic_topic = declare_parameter<std::string>("dynamic_joint_states_topic", "");
    auto imu_topic = declare_parameter<std::string>("imu_topic", "/policy_probe/imu");
    auto permission_topic = declare_parameter<std::string>("permission_topic", "/policy_probe/permission");
    command_topic_ = declare_parameter<std::string>("command_topic", "/policy_probe/commands");
    traced_topic_ = declare_parameter<std::string>("traced_command_topic", "/policy_probe/commands_traced");
    event_topic_ = declare_parameter<std::string>("trace_event_topic", "/policy_probe/events");
    metric_topic_ = declare_parameter<std::string>("metrics_topic", "/policy_probe/metrics");
    candidate_topic_ = declare_parameter<std::string>("candidate_topic", "/policy_probe/execution/policy/commands");
    required_ = declare_parameter<bool>("operator_input_required", false);
    use_sim_ = get_parameter("use_sim_time").as_bool();
    const std::string domain = std::getenv("ROS_DOMAIN_ID");
    const auto profile_env = std::getenv("MARVIN_POLICY_NATIVE_PROFILE");
    physical_ = profile_env && std::string(profile_env) == "marvin-v1";
    std::map<std::string, std::string> topics{
      {"joint_states_topic", joint_topic}, {"dynamic_joint_states_topic", dynamic_topic},
      {"imu_topic", imu_topic}, {"permission_topic", permission_topic},
      {"command_topic", command_topic_}, {"traced_command_topic", traced_topic_},
      {"trace_event_topic", event_topic_}, {"metrics_topic", metric_topic_},
      {"candidate_topic", candidate_topic_}};
    for (auto &entry : topics)
      if (!entry.second.empty()) entry.second = get_node_topics_interface()->resolve_topic_name(entry.second);
    mio::validate_profile(domain, profile_env ? profile_env : "test", get_namespace(),
                          use_sim_, required_, names_, topics);
    const auto qos = rclcpp::SensorDataQoS().keep_last(1);
    if (dynamic_topic.empty()) {
      joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(joint_topic, qos,
          [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) { joint(*msg); });
    } else {
      dynamic_sub_ = create_subscription<control_msgs::msg::DynamicJointState>(dynamic_topic, qos,
          [this](control_msgs::msg::DynamicJointState::ConstSharedPtr msg) { dynamic(*msg); });
    }
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(imu_topic, qos,
        [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) {
          imu_valid_ = false;
          const auto &q = msg->orientation; const auto &w = msg->angular_velocity;
          const auto &a = msg->linear_acceleration;
          const std::array<double, 10> values{q.x,q.y,q.z,q.w,w.x,w.y,w.z,a.x,a.y,a.z};
          for (auto v : values) if (!std::isfinite(v)) return;
          std::copy(values.begin(), values.end(), data_.begin() + 24);
          imu_stamp_ = rclcpp::Time(msg->header.stamp).nanoseconds();
          imu_receipt_ = steady_ns(); imu_valid_ = true; ++imu_count_;
        });
    permission_sub_ = create_subscription<std_msgs::msg::Bool>(permission_topic, 1,
        [this](std_msgs::msg::Bool::ConstSharedPtr msg) {
          permission_ = msg->data; permission_receipt_ = steady_ns();
        });
    ros_timer_ = rclcpp::create_timer(this, get_clock(), std::chrono::milliseconds(5),
                                    [this] { tick(); });
    // Management and parent-death checks remain live when simulation is paused.
    wall_timer_ = create_wall_timer(std::chrono::milliseconds(20), [this] { tick(); });
  }

 private:
  bool fresh(uint64_t now, uint64_t stamp, uint64_t age) const {
    return now >= stamp && now - stamp <= age;
  }
  void joint(const sensor_msgs::msg::JointState &msg) {
    joint_valid_ = false;
    if (std::set<std::string>(msg.name.begin(), msg.name.end()).size() != msg.name.size()) return;
    for (size_t i = 0; i < 12; ++i) {
      auto found = std::find(msg.name.begin(), msg.name.end(), names_[i]);
      auto index = size_t(std::distance(msg.name.begin(), found));
      if (found == msg.name.end() || index >= msg.position.size() || index >= msg.velocity.size() ||
          !std::isfinite(msg.position[index]) || !std::isfinite(msg.velocity[index])) return;
      data_[i] = msg.position[index]; data_[12 + i] = msg.velocity[index];
    }
    joint_stamp_ = rclcpp::Time(msg.header.stamp).nanoseconds();
    joint_receipt_ = steady_ns(); joint_valid_ = true; ++joint_count_;
  }
  void dynamic(const control_msgs::msg::DynamicJointState &msg) {
    joint_valid_ = false;
    if (msg.joint_names.size() != msg.interface_values.size() ||
        std::set<std::string>(msg.joint_names.begin(), msg.joint_names.end()).size() != msg.joint_names.size()) return;
    for (size_t i = 0; i < 12; ++i) {
      auto found = std::find(msg.joint_names.begin(), msg.joint_names.end(), names_[i]);
      if (found == msg.joint_names.end()) return;
      const auto &row = msg.interface_values[std::distance(msg.joint_names.begin(), found)];
      if (row.interface_names.size() != row.values.size() ||
          std::set<std::string>(row.interface_names.begin(), row.interface_names.end()).size() != row.interface_names.size()) return;
      for (size_t field = 0; field < 2; ++field) {
        auto name = field == 0 ? "position" : "velocity";
        auto where = std::find(row.interface_names.begin(), row.interface_names.end(), name);
        if (where == row.interface_names.end()) return;
        double value = row.values[std::distance(row.interface_names.begin(), where)];
        if (!std::isfinite(value)) return;
        data_[field * 12 + i] = value;
      }
      auto receipt = std::find(row.interface_names.begin(), row.interface_names.end(), "encoder_receipt_monotonic_s");
      double value = receipt == row.interface_names.end() ? 0. : row.values[std::distance(row.interface_names.begin(), receipt)];
      encoders_[i] = std::isfinite(value) && value > 0. ? value : 0.;
    }
    joint_stamp_ = rclcpp::Time(msg.header.stamp).nanoseconds();
    joint_receipt_ = steady_ns(); joint_valid_ = true; ++joint_count_;
  }
  bool send_packet(const Packet &packet) {
    auto bytes = mio::encode(packet);
    if (::send(fd_, bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_NOSIGNAL) != ssize_t(bytes.size())) {
      gate_.fail(mio::Fault::transport_closed); if (ramp_) ramp_->stop();
      rclcpp::shutdown(); return false;
    }
    return true;
  }
  uint64_t flags() const {
    return (ramp_ && (gateway_mode_ ? gateway_.released() : ramp_->output_released()) ? uint64_t(Flags::output_released) : 0) |
           (gate_.movement_permitted() ? uint64_t(Flags::permitted) : 0) |
           (gate_.enabled() ? uint64_t(Flags::active) : 0) |
           (sensors_ ? uint64_t(Flags::sensors_valid) : 0) |
           (gate_.paused() ? uint64_t(Flags::paused) : 0) |
           (gate_.fault() != mio::Fault::none ? uint64_t(Flags::faulted) : 0) |
           (last_output_valid_ ? uint64_t(Flags::output_valid) : 0) |
           (dynamic_sub_ ? uint64_t(Flags::encoder_timing) : 0);
  }
  void ack(const Packet &request, bool success, uint64_t ros) {
    Packet packet; packet.kind = Kind::ack; packet.sequence = request.sequence;
    packet.generation = gate_.generation(); packet.ros_ns = ros;
    packet.flags = success ? uint64_t(Flags::success) : 0;
    send_packet(packet);
  }
  void configure(const Packet &p) {
    if (output_ || p.values.size() != 43) throw std::invalid_argument("Invalid/repeated native configuration");
    mio::OutputConfig config;
    std::copy_n(p.values.begin(), 12, config.defaults.begin());
    std::copy_n(p.values.begin() + 12, 12, config.lower.begin());
    std::copy_n(p.values.begin() + 24, 12, config.upper.begin());
    config.scale = p.values[36]; config.smooth = p.values[39] == 1.;
    if (p.values[39] != 0. && p.values[39] != 1.) throw std::invalid_argument("Invalid mapping mode");
    if (p.values[40] < 1. || p.values[40] > 1000. || p.values[41] != 5. ||
        p.values[42] <= 0. || p.values[42] > 300.) throw std::invalid_argument("Invalid native timing configuration");
    if (physical_ && p.values[40] != 100.) throw std::invalid_argument("marvin-v1 requires 100 ms inference expiry");
    gateway_mode_ = p.flags & 16;
    if (gateway_mode_ && (p.flags & 3)) throw std::invalid_argument("Gateway and direct output are exclusive");
    if (physical_ && !gateway_mode_ && (p.flags & 3) != 2) throw std::invalid_argument("marvin-v1 requires traced commands only");
    gate_ = mio::Lifecycle(20000000, uint64_t(p.values[40] * 1e6));
    output_ = std::make_unique<mio::ActionOutput>(config);
    ramp_ = std::make_unique<mio::Ramp>(p.values[37], p.values[38]);
    const auto qos = rclcpp::SensorDataQoS().keep_last(1);
    if (p.flags & 1) legacy_ = create_publisher<std_msgs::msg::Float64MultiArray>(command_topic_, qos);
    if (p.flags & 2) traced_ = create_publisher<marvin_trace_msgs::msg::TracedFloat64MultiArray>(traced_topic_, qos);
    if (gateway_mode_) candidate_ = create_publisher<runtime_interfaces::msg::ExecutorCommand>(candidate_topic_, qos);
    if (!legacy_ && !traced_ && !candidate_) throw std::invalid_argument("No command output selected");
    if (p.flags & 4) metrics_ = create_publisher<std_msgs::msg::String>(metric_topic_, 10);
    if (p.flags & 8) events_ = create_publisher<marvin_trace_msgs::msg::TraceEvent>(event_topic_, qos);
    if (p.flags & ~uint64_t(31)) throw std::invalid_argument("Unknown configuration flags");
    trace_prefix_ = p.trace_id & 0xffffff0000000000ULL;
    capture_until_ = steady_ns() + uint64_t(p.values[42] * 1e9);
  }
  void incoming(uint64_t mono, uint64_t ros) {
    for (int n = 0; n < 16; ++n) {
      std::array<uint8_t, mio::max_packet> bytes{};
      auto count = recv(fd_, bytes.data(), bytes.size(), MSG_DONTWAIT | MSG_TRUNC);
      if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
      if (count <= 0) { gate_.fail(mio::Fault::transport_closed); if (ramp_) ramp_->stop(); rclcpp::shutdown(); return; }
      const auto p = mio::decode(bytes.data(), size_t(count));
      last_peer_ = mono;
      if (p.kind == Kind::configure) { configure(p); ack(p, true, ros); continue; }
      if (!output_) throw std::invalid_argument("Native configuration required");
      if (p.kind == Kind::status) continue; // Parent heartbeat; never grants ownership.
      if (p.kind == Kind::prepare) {
        const bool ok = gateway_mode_ && gateway_.prepare(p, mono, !gate_.enabled() && ramp_->output_released());
        if (ok) gate_.stop(); // Fresh preparation clears an inactive fault, never grants output.
        ack(p, ok, ros); continue;
      }
      if (p.kind == Kind::shutdown) { gate_.stop(); gateway_.stop(); ramp_->stop(); ack(p, true, ros); rclcpp::shutdown(); return; }
      if (p.kind == Kind::activate) {
        if (!sensors_ || (gateway_mode_ && !gateway_.ready(mono))) { ack(p, false, ros); continue; }
        gate_.activate(); ramp_->activate(ros / 1e9); pending_commit_.reset();
        if (gateway_mode_) gateway_.start(gate_.generation(), ros);
        ack(p, true, ros); continue;
      }
      if (p.kind == Kind::deactivate) {
        gate_.stop(); pending_commit_.reset();
        gateway_.stop();
        if (gateway_mode_ || (p.flags & 1)) ramp_->stop(); else ramp_->deactivate(ros / 1e9, action_);
        ack(p, true, ros); continue;
      }
      if (p.kind == Kind::invalidate) {
        gate_.invalidate(); gateway_.rebind(gate_.generation()); pending_commit_.reset(); ack(p, true, ros); continue;
      }
      if (p.kind == Kind::fault) { gateway_.stop(); gate_.fail(mio::Fault::invalid_result); ramp_->stop(); pending_commit_.reset(); continue; }
      if (p.kind != Kind::result || p.values.size() != 24 ||
          (p.flags & ~uint64_t(Flags::output_locked | Flags::direct_output)))
        throw std::invalid_argument("Invalid native result");
      mio::Joints candidate{}, locked{};
      std::copy_n(p.values.begin(), 12, candidate.begin());
      std::copy_n(p.values.begin() + 12, 12, locked.begin());
      if (p.flags & Flags::output_locked) output_->lock(locked);
      if (p.flags & Flags::direct_output) output_->lock(candidate);
      else for (auto value : candidate) if (value < -10. || value > 10.)
        throw std::invalid_argument("Raw action exceeds runner bounds");
      if (gate_.accept({p.sequence, p.generation, p.requested_ns, p.ros_ns}, mono, ros)) {
        action_ = candidate; locked_ = locked; result_flags_ = p.flags;
        pending_commit_ = p; ++accepted_count_;
      }
    }
  }
  uint64_t next_trace() {
    trace_counter_ = (trace_counter_ + 1) & 0xffffffffffULL;
    if (!trace_counter_) ++trace_counter_;
    return trace_prefix_ | trace_counter_;
  }
  void publish(mio::Joints policy, uint64_t mono, uint64_t ros) {
    auto actual = result_flags_ & Flags::output_locked ? locked_ : policy;
    if (gateway_mode_) {
      actual = gateway_.command(actual, gate_.generation(), ros);
    }
    auto id = pending_commit_ ? pending_commit_->trace_id : next_trace();
    if (candidate_) { runtime_interfaces::msg::ExecutorCommand msg;
      msg.stamp = rclcpp::Time(int64_t(ros)); msg.trace_id = id;
      msg.generation = gateway_.token(); msg.sequence = gateway_.sequence();
      msg.data.assign(actual.begin(), actual.end()); candidate_->publish(msg);
    }
    if (legacy_) { std_msgs::msg::Float64MultiArray msg; msg.data.assign(actual.begin(), actual.end()); legacy_->publish(msg); }
    if (traced_) { marvin_trace_msgs::msg::TracedFloat64MultiArray msg;
      msg.stamp = rclcpp::Time(int64_t(ros)); msg.trace_id = id;
      msg.data.assign(actual.begin(), actual.end()); traced_->publish(msg); }
    if (events_ && mono < capture_until_) { marvin_trace_msgs::msg::TraceEvent event;
      event.stamp = rclcpp::Time(int64_t(ros)); event.trace_id = id;
      event.stage = gateway_mode_ ? "policy_candidate_published" : "command_published"; event.source = "marvin_policy_io"; events_->publish(event); }
    last_policy_ = policy; last_output_valid_ = true; ++command_count_;
    if (last_publish_) max_gap_ns_ = std::max(max_gap_ns_, mono - last_publish_);
    last_publish_ = mono;
    if (pending_commit_) {
      auto commit = *pending_commit_; commit.kind = Kind::commit;
      commit.values.assign(policy.begin(), policy.end());
      commit.values.insert(commit.values.end(), actual.begin(), actual.end());
      send_packet(commit); pending_commit_.reset();
    }
  }
  Packet snapshot(Kind kind, uint64_t mono, uint64_t ros) {
    Packet packet; packet.kind = kind; packet.generation = gate_.generation();
    packet.requested_ns = mono; packet.ros_ns = ros;
    packet.joint_ns = joint_stamp_; packet.imu_ns = imu_stamp_; packet.flags = flags();
    if (kind == Kind::status) packet.trace_id = uint64_t(gate_.fault());
    packet.values.assign(data_.begin(), data_.end());
    packet.values.insert(packet.values.end(), last_policy_.begin(), last_policy_.end());
    packet.values.insert(packet.values.end(), encoders_.begin(), encoders_.end());
    return packet;
  }
  void metric(uint64_t mono) {
    if (!metrics_ || mono - metric_start_ < 1000000000ULL) return;
    const double seconds = (mono - metric_start_) / 1e9;
    std::ostringstream json;
    json << "{\"schema_version\":\"runtime-timing.v1\",\"source\":\"policy_native_io\",\"window_s\":" << seconds
         << ",\"counts\":{\"commands\":" << command_count_ << ",\"accepted\":" << accepted_count_
         << ",\"joint_callbacks\":" << joint_count_ << ",\"imu_callbacks\":" << imu_count_ << "},\"rates_hz\":{\"commands\":"
         << command_count_ / seconds << ",\"accepted\":" << accepted_count_ / seconds
         << "},\"command_gap_max_ms\":" << max_gap_ns_ / 1e6 << ",\"generation\":" << gate_.generation()
         << ",\"fault\":" << int(gate_.fault()) << "}";
    std_msgs::msg::String msg; msg.data = json.str(); metrics_->publish(msg);
    command_count_ = accepted_count_ = joint_count_ = imu_count_ = max_gap_ns_ = 0;
    metric_start_ = mono;
  }
  void tick() {
    try {
      const auto mono = steady_ns(); const auto ros = uint64_t(now().nanoseconds());
      const auto tick_gap = last_tick_ && mono >= last_tick_ ? mono - last_tick_ : 0;
      last_tick_ = mono;
      sensors_ = joint_valid_ && imu_valid_ && fresh(ros, joint_stamp_, 500000000) && fresh(ros, imu_stamp_, 500000000) &&
                 fresh(mono, joint_receipt_, 500000000) && fresh(mono, imu_receipt_, 500000000);
      const auto generation = gate_.generation();
      gate_.observe(mono, ros, sensors_, !required_ || (permission_ && fresh(mono, permission_receipt_, 300000000)));
      if (last_ros_ && ros < *last_ros_) {
        last_output_valid_ = false; next_publish_ = ros;
        if (gateway_mode_) { gate_.stop(); gateway_.stop(); }
        if (ramp_) { ramp_->stop(); if (gate_.enabled()) ramp_->activate(ros / 1e9); }
      }
      last_ros_ = ros;
      if (generation != gate_.generation()) {
        pending_commit_.reset();
        // Permission/sensor/override invalidation discards inference, not supervisor ownership.
        gateway_.rebind(gate_.generation());
      }
      incoming(mono, ros);
      // Apply current safety state before accepting queued results. Enforce the
      // deadline after the drain so an already-arrived fresh result can replace
      // an expired accepted result before any command publication.
      gate_.enforce_deadlines(mono, ros);
      if (!rclcpp::ok()) return;
      if (!output_) {
        if (mono - boot_ > 10000000000ULL) throw std::runtime_error("Native configuration timeout");
        return;
      }
      // Idle startup has no command output to protect. Keep the heartbeat
      // deadline while active or ramping down; never bypass it during motion.
      if ((gate_.enabled() || !ramp_->output_released()) &&
          mono - last_peer_ > 300000000ULL) { gate_.fail(mio::Fault::transport_closed); ramp_->stop(); }
      if (gate_.fault() != mio::Fault::none) { gateway_.stop(); ramp_->stop(); pending_commit_.reset(); }
      if (gate_.fault() != last_fault_) {
        last_fault_ = gate_.fault();
        if (last_fault_ != mio::Fault::none) {
          RCLCPP_ERROR(get_logger(), "Native output fault=%d; explicit reactivation required", int(last_fault_));
          if (last_fault_ == mio::Fault::transport_closed) {
            // Record only on a fault transition. These are native-loop ages,
            // not Python callback timings or proof of a particular stall cause.
            std::ostringstream json;
            json << "{\"generation\":" << gate_.generation()
                 << ",\"limit_ms\":300,\"peer_age_ms\":" << (mono - last_peer_) / 1e6
                 << ",\"tick_gap_ms\":" << tick_gap / 1e6
                 << ",\"status_send_age_ms\":";
            if (last_status_) json << (mono - last_status_) / 1e6;
            else json << "null";
            json << "}";
            RCLCPP_ERROR(get_logger(), "NATIVE_TRANSPORT_FAULT %s", json.str().c_str());
          }
          if (last_fault_ == mio::Fault::inference_expired && gate_.expiry()) {
            const auto &d = *gate_.expiry();
            std::ostringstream json;
            json << "{\"branch\":\"" << d.branch << "\",\"generation\":" << d.generation
                 << ",\"limit_ms\":" << d.max_age_ns / 1e6
                 << ",\"tick_gap_ms\":" << tick_gap / 1e6;
            auto request = [&](const char *name, const std::optional<mio::Request> &r) {
              json << ",\"" << name << "\":";
              if (!r) { json << "null"; return; }
              json << "{\"sequence\":" << r->sequence << ",\"steady_age_ms\":"
                   << (double(d.steady_ns) - double(r->steady_ns)) / 1e6
                   << ",\"ros_age_ms\":" << (double(d.ros_ns) - double(r->ros_ns)) / 1e6 << "}";
            };
            request("pending", d.pending); request("accepted", d.accepted);
            json << "}";
            RCLCPP_ERROR(get_logger(), "NATIVE_DEADLINE_FAULT %s", json.str().c_str());
          }
        }
      }
      if (ros >= next_publish_) {
        if (gate_.can_publish(mono, ros)) {
          auto policy = result_flags_ & Flags::direct_output ? action_ : output_->map(action_, ramp_->activation_factor(ros / 1e9));
          publish(policy, mono, ros);
        } else if (!gateway_mode_ && !gate_.enabled() && gate_.fault() == mio::Fault::none && sensors_ && !gate_.paused()) {
          if (auto action = ramp_->deactivation_action(ros / 1e9)) publish(output_->map(*action, 1.), mono, ros);
        }
        const uint64_t period = gateway_mode_ ? 20000000 : 5000000;
        next_publish_ += ((ros - next_publish_) / period + 1) * period;
      }
      if (auto request = pending_commit_ ? std::nullopt : gate_.request(mono, ros)) {
        auto packet = snapshot(Kind::snapshot, mono, ros);
        packet.sequence = request->sequence; packet.trace_id = next_trace(); send_packet(packet);
      }
      if (mono - last_status_ >= 50000000ULL) {
        send_packet(snapshot(Kind::status, mono, ros)); last_status_ = mono;
      }
      metric(mono);
    } catch (const std::exception &e) {
      gate_.fail(mio::Fault::invalid_result); if (ramp_) ramp_->stop();
      RCLCPP_ERROR(get_logger(), "Native I/O stopped: %s", e.what()); rclcpp::shutdown();
    }
  }

  int fd_; std::vector<std::string> names_;
  uint64_t boot_, metric_start_, last_peer_ = 0, last_status_ = 0, next_publish_ = 0;
  uint64_t last_tick_ = 0;
  uint64_t joint_stamp_ = 0, imu_stamp_ = 0, joint_receipt_ = 0, imu_receipt_ = 0, permission_receipt_ = 0;
  uint64_t trace_prefix_ = 0, trace_counter_ = 0, capture_until_ = 0, result_flags_ = 0;
  uint64_t command_count_ = 0, accepted_count_ = 0, joint_count_ = 0, imu_count_ = 0, last_publish_ = 0, max_gap_ns_ = 0;
  bool required_ = false, permission_ = false, sensors_ = false, use_sim_ = false, physical_ = false;
  bool joint_valid_ = false, imu_valid_ = false, last_output_valid_ = false;
  std::array<double,34> data_{};
  mio::Joints encoders_{}, action_{}, locked_{}, last_policy_{};
  std::optional<uint64_t> last_ros_;
  std::optional<Packet> pending_commit_;
  mio::Lifecycle gate_;
  mio::GatewayOutput gateway_;
  bool gateway_mode_ = false;
  mio::Fault last_fault_ = mio::Fault::none;
  std::unique_ptr<mio::ActionOutput> output_;
  std::unique_ptr<mio::Ramp> ramp_;
  std::string command_topic_, traced_topic_, event_topic_, metric_topic_, candidate_topic_;
  rclcpp::Publisher<runtime_interfaces::msg::ExecutorCommand>::SharedPtr candidate_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<control_msgs::msg::DynamicJointState>::SharedPtr dynamic_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr permission_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr legacy_;
  rclcpp::Publisher<marvin_trace_msgs::msg::TracedFloat64MultiArray>::SharedPtr traced_;
  rclcpp::Publisher<marvin_trace_msgs::msg::TraceEvent>::SharedPtr events_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr metrics_;
  rclcpp::TimerBase::SharedPtr ros_timer_, wall_timer_;
};

int main(int argc, char **argv) {
  try {
    auto domain = std::getenv("ROS_DOMAIN_ID"), local = std::getenv("ROS_LOCALHOST_ONLY");
    const auto profile = std::getenv("MARVIN_POLICY_NATIVE_PROFILE");
    mio::validate_environment(domain ? domain : "", local ? local : "", profile ? profile : "test");
    if (argc < 2) throw std::runtime_error("Native I/O requires an inherited socket argument");
    int fd = std::stoi(argv[1]), type = 0; socklen_t size = sizeof(type);
    if (getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &size) || type != SOCK_SEQPACKET)
      throw std::runtime_error("Native I/O requires inherited sequence-packet socket");
    sockaddr_storage address{}; size = sizeof(address);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&address), &size) || address.ss_family != AF_UNIX)
      throw std::runtime_error("Native I/O socket must be local");
    rclcpp::init(argc - 1, argv + 1);
    auto node = std::make_shared<NativeIo>(fd); rclcpp::spin(node);
    node.reset(); if (rclcpp::ok()) rclcpp::shutdown(); close(fd); return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
