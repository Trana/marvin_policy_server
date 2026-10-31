// Offline contract check. This executable neither initializes ROS nor opens IPC.
#include "profile.hpp"
#include <iostream>

int main(int argc, char **argv) {
  try {
    if (argc != 28) throw std::invalid_argument("Expected 27 profile fields");
    marvin::native_io::validate_environment(argv[1], argv[2], argv[3]);
    const std::vector<std::string> keys{"joint_states_topic", "dynamic_joint_states_topic",
      "imu_topic", "permission_topic", "command_topic", "traced_command_topic",
      "trace_event_topic", "metrics_topic", "candidate_topic"};
    std::map<std::string, std::string> topics;
    for (size_t i = 0; i < keys.size(); ++i) topics[keys[i]] = argv[7 + i];
    std::vector<std::string> names(argv + 16, argv + 28);
    marvin::native_io::validate_profile(argv[1], argv[3], argv[4],
                                       std::string(argv[5]) == "true",
                                       std::string(argv[6]) == "true", names, topics);
    return 0;
  } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
