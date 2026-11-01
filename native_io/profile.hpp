#pragma once
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace marvin::native_io {
inline void validate_environment(const std::string &domain, const std::string &local,
                                 const std::string &profile) {
  if (local != "1") throw std::invalid_argument("Native I/O requires localhost-only discovery");
  if (profile == "test" && (domain == "10" || domain == "184")) return;
  if (profile == "marvin-v1" && domain == "0") return;
  throw std::invalid_argument("Native I/O requires an explicit matching deployment profile");
}

inline void validate_profile(const std::string &domain, const std::string &profile,
    const std::string &ns, bool simulated, bool required, const std::vector<std::string> &names,
    const std::map<std::string, std::string> &topics) {
  if (profile == "marvin-v1") {
    if (domain != "0" || ns != "/" || simulated || !required)
      throw std::invalid_argument("marvin-v1 requires domain 0, root namespace, wall time and operator permission");
    std::vector<std::string> expected_names;
    for (const auto &axis : {"hip", "thigh", "calf"})
      for (const auto &leg : {"FL", "RL", "FR", "RR"})
        expected_names.push_back(std::string(leg) + "_" + axis + "_joint");
    if (names != expected_names) throw std::invalid_argument("marvin-v1 requires physical controller joint order");
    const std::map<std::string, std::string> expected_topics{
      {"joint_states_topic", "/joint_states"}, {"dynamic_joint_states_topic", ""},
      {"imu_topic", "/imu"}, {"permission_topic", "/operator/motion_permitted"},
      {"command_topic", "/marvin_joint_controller/commands"},
      {"traced_command_topic", "/marvin_joint_controller/commands_traced"},
      {"trace_event_topic", "/trace/events"}, {"metrics_topic", "/trace/metrics"},
      {"candidate_topic", "/execution/policy/commands"}};
    if (topics != expected_topics) throw std::invalid_argument("marvin-v1 topic remapping is not allowed");
  } else if (profile == "test" && domain == "10") {
    if (!simulated || ns != "/sim") throw std::invalid_argument("Domain 10 requires simulation time and /sim namespace");
  } else if (profile == "test" && domain == "184") {
    for (const auto &entry : topics)
      if (!entry.second.empty() && entry.second.rfind("/policy_probe/", 0) != 0)
        throw std::invalid_argument("Isolated domain requires /policy_probe/ topics");
  } else {
    throw std::invalid_argument("Unknown native deployment profile");
  }
}
}  // namespace marvin::native_io
