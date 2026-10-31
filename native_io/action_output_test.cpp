#include "action_output.hpp"
#include <iomanip>
#include <iostream>
#include <limits>

using namespace marvin::native_io;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (false)

int main(int argc, char **) {
  if (argc == 2) {
    // Test-only streaming calculator for Python/C++ numerical equivalence.
    // Each line: smooth scale factor, twelve default/lower/upper/action tuples.
    OutputConfig cfg; double factor; Joints action{};
    std::cout << std::setprecision(17);
    while (std::cin >> cfg.smooth >> cfg.scale >> factor) {
      for (size_t i = 0; i < 12; ++i)
        if (!(std::cin >> cfg.defaults[i] >> cfg.lower[i] >> cfg.upper[i] >> action[i])) return 2;
      for (auto v : ActionOutput(cfg).map(action, factor)) std::cout << v << ' ';
      std::cout << '\n';
    }
    return 0;
  }
  OutputConfig cfg; cfg.lower.fill(-1.); cfg.upper.fill(1.);
  ActionOutput output(cfg); Joints action{}; action.fill(8.);
  CHECK(output.map(action, .25)[0] == .5);
  CHECK(output.map(action, 1.)[0] == 1.);
  CHECK(output.map(action, 0.)[0] == 0.);
  bool invalid = false;
  try { output.lock(action); } catch (const std::invalid_argument &) { invalid = true; }
  CHECK(invalid);
  action.fill(.5); CHECK(output.lock(action) == action);
  Ramp ramp(1., 1.);
  CHECK(ramp.output_released());
  CHECK(!ramp.active()); CHECK(!ramp.deactivation_action(1.));
  ramp.activate(1.); CHECK(ramp.activation_factor(1.) == 0.);
  CHECK(ramp.activation_factor(1.5) == .5);
  ramp.activate(1.5); CHECK(ramp.activation_factor(1.75) == .75);
  CHECK(ramp.activation_factor(2.) == 1.);
  ramp.deactivate(2., action); CHECK(!ramp.active());
  CHECK(!ramp.output_released());
  CHECK(ramp.deactivation_action(2.5)->at(0) == .25);
  CHECK(!ramp.deactivation_action(3.));
  CHECK(ramp.output_released());
  ramp.activate(4.); ramp.stop(); CHECK(!ramp.active());
  CHECK(ramp.activation_factor(4.) == 0.);
  Ramp immediate(0., 0.); immediate.activate(0.);
  CHECK(immediate.activation_factor(0.) == 1.);
  immediate.deactivate(0., action); CHECK(!immediate.deactivation_action(0.));
  {
    // Characterize the existing takeover, not an acceptance criterion for smooth motion.
    // Saved sim trial: FL calf was 1.4595 rad just before the first policy target.
    OutputConfig takeover;
    takeover.lower.fill(-2.); takeover.upper.fill(2.);
    takeover.defaults[8] = 0.6981317007977318;
    Joints raw{}; raw.fill(.5);
    for (bool smooth : {false, true}) {
      takeover.smooth = smooth;
      ActionOutput mapper(takeover);
      Ramp activation(1., 1.); activation.activate(65.39);
      const auto first = mapper.map(raw, activation.activation_factor(65.39));
      CHECK(first[8] == takeover.defaults[8]);
      CHECK(std::abs(first[8] - 1.4595) > .76);
    }
  }
  invalid = false; cfg.scale = std::numeric_limits<double>::quiet_NaN();
  try { ActionOutput bad(cfg); } catch (const std::invalid_argument &) { invalid = true; }
  CHECK(invalid);
  std::cout << "Native output checks passed\n";
}
