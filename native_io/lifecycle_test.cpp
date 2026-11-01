#include "lifecycle.hpp"
#include <iostream>

using namespace marvin::native_io;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (false)
constexpr uint64_t ms = 1000000;

Lifecycle started() {
  Lifecycle gate;
  CHECK(!gate.enabled());
  gate.observe(1 * ms, 1 * ms, true, true);
  CHECK(!gate.request(1 * ms, 1 * ms));
  gate.activate();
  return gate;
}

void tick(Lifecycle &gate, uint64_t steady_ns, uint64_t ros_ns,
          bool sensors_valid, bool movement_permitted) {
  gate.observe(steady_ns, ros_ns, sensors_valid, movement_permitted);
  gate.enforce_deadlines(steady_ns, ros_ns);
}

int main() {
  {
    auto gate = started();
    auto old = gate.request(ms, ms); CHECK(old);
    CHECK(gate.accept(*old, 2 * ms, 2 * ms));
    tick(gate, 81 * ms, 81 * ms, true, true);
    auto pending = gate.request(81 * ms, 81 * ms); CHECK(pending);
    tick(gate, 101 * ms, 101 * ms, true, true);
    CHECK(gate.enabled()); CHECK(!gate.expiry()); // Exactly 100 ms remains valid.
    gate.observe(102 * ms, 102 * ms, true, true);
    CHECK(!gate.can_publish(102 * ms, 102 * ms)); // Never publish the stale result between phases.
    CHECK(gate.accept(*pending, 102 * ms, 102 * ms));
    gate.enforce_deadlines(102 * ms, 102 * ms);
    CHECK(gate.enabled()); CHECK(!gate.expiry());
    CHECK(gate.can_publish(102 * ms, 102 * ms));
  }
  {
    auto gate = started();
    auto old = gate.request(ms, ms); CHECK(old);
    CHECK(gate.accept(*old, 2 * ms, 2 * ms));
    gate.observe(102 * ms, 102 * ms, true, true);
    CHECK(gate.enabled()); CHECK(!gate.expiry());
    CHECK(!gate.can_publish(102 * ms, 102 * ms));
    gate.enforce_deadlines(102 * ms, 102 * ms);
    CHECK(std::string(gate.expiry()->branch) == "accepted");
    CHECK(!gate.expiry()->pending);
    CHECK(gate.expiry()->accepted->sequence == old->sequence);
    CHECK(gate.expiry()->generation == old->generation);
  }
  {
    auto gate = started();
    auto request = gate.request(ms, ms); CHECK(request);
    CHECK(!gate.request(ms, ms));  // One request only.
    CHECK(!gate.can_publish(ms, ms));
    auto wrong = *request; ++wrong.sequence;
    CHECK(!gate.accept(wrong, 2 * ms, 2 * ms));
    wrong = *request; ++wrong.generation;
    CHECK(!gate.accept(wrong, 2 * ms, 2 * ms));
    wrong = *request; ++wrong.steady_ns;
    CHECK(!gate.accept(wrong, 2 * ms, 2 * ms));
    wrong = *request; ++wrong.ros_ns;
    CHECK(!gate.accept(wrong, 2 * ms, 2 * ms));
    CHECK(gate.accept(*request, 2 * ms, 2 * ms));
    CHECK(!gate.accept(*request, 2 * ms, 2 * ms));
    CHECK(gate.can_publish(2 * ms, 2 * ms));
    gate.stop();
    CHECK(!gate.can_publish(3 * ms, 3 * ms));
    gate.activate(); CHECK(!gate.accept(*request, 3 * ms, 3 * ms));
  }
  {
    auto gate = started();
    auto before = gate.request(ms, ms); CHECK(before);
    tick(gate, 5 * ms, 5 * ms, true, false);
    CHECK(gate.enabled()); CHECK(!gate.movement_permitted());
    CHECK(!gate.accept(*before, 6 * ms, 6 * ms));
    auto neutral = gate.request(6 * ms, 6 * ms); CHECK(neutral);
    CHECK(gate.accept(*neutral, 7 * ms, 7 * ms));
    CHECK(gate.can_publish(7 * ms, 7 * ms));  // Balance remains possible.
    tick(gate, 8 * ms, 8 * ms, true, true);
    CHECK(!gate.can_publish(8 * ms, 8 * ms));
    CHECK(gate.request(8 * ms, 8 * ms));
  }
  {
    auto gate = started();
    auto request = gate.request(ms, ms); CHECK(request);
    tick(gate, 2 * ms, 2 * ms, false, true);
    CHECK(!gate.accept(*request, 3 * ms, 3 * ms));
    CHECK(!gate.request(3 * ms, 3 * ms));
    tick(gate, 4 * ms, 4 * ms, true, true);
    CHECK(gate.request(4 * ms, 4 * ms));
  }
  {
    auto gate = started();
    auto request = gate.request(ms, ms); CHECK(request);
    CHECK(!gate.accept(*request, 102 * ms, 2 * ms)); // Wall time expiry.
    CHECK(gate.fault() == Fault::inference_expired);
    CHECK(gate.expiry());
    CHECK(std::string(gate.expiry()->branch) == "arriving_result");
    CHECK(gate.expiry()->pending->sequence == request->sequence);
    CHECK(gate.expiry()->steady_ns - gate.expiry()->pending->steady_ns == 101 * ms);
    tick(gate, 103 * ms, 3 * ms, true, true);
    CHECK(!gate.request(103 * ms, 3 * ms)); // No automatic restart.
    gate.activate(); CHECK(gate.request(103 * ms, 3 * ms));
    CHECK(!gate.expiry());
  }
  {
    auto gate = started();
    CHECK(gate.request(ms, ms));
    tick(gate, 102 * ms, 102 * ms, true, true);
    CHECK(!gate.enabled()); CHECK(gate.fault() == Fault::inference_expired);
    CHECK(std::string(gate.expiry()->branch) == "pending");
    CHECK(!gate.expiry()->accepted);
  }
  {
    auto gate = started();
    auto request = gate.request(ms, ms); CHECK(request);
    CHECK(gate.accept(*request, 2 * ms, 2 * ms));
    tick(gate, 102 * ms, 102 * ms, true, true);
    CHECK(!gate.can_publish(102 * ms, 102 * ms)); // Held action expires too.
    CHECK(std::string(gate.expiry()->branch) == "accepted");
    CHECK(gate.expiry()->accepted->sequence == request->sequence);
    CHECK(!gate.expiry()->pending);
    gate.stop(); CHECK(!gate.expiry());
  }
  {
    auto gate = started();
    auto request = gate.request(ms, ms); CHECK(request);
    tick(gate, 2 * ms, ms, true, true);
    CHECK(!gate.can_publish(2 * ms, ms)); // No commands while ROS time is unchanged.
    tick(gate, 101 * ms, ms, true, true);
    CHECK(gate.paused()); CHECK(gate.enabled());
    CHECK(!gate.accept(*request, 101 * ms, ms));
    tick(gate, 500 * ms, ms, true, true);
    CHECK(gate.fault() == Fault::none);
    tick(gate, 501 * ms, 2 * ms, true, true);
    CHECK(!gate.paused()); CHECK(!gate.can_publish(501 * ms, 2 * ms));
    auto resumed = gate.request(501 * ms, 2 * ms); CHECK(resumed);
    CHECK(gate.accept(*resumed, 502 * ms, 3 * ms));
    CHECK(gate.can_publish(502 * ms, 3 * ms));
  }
  {
    auto gate = started();
    auto before = gate.request(ms, ms); CHECK(before);
    tick(gate, 2 * ms, 0, true, true); // Simulator reset to zero.
    CHECK(!gate.accept(*before, 2 * ms, 0));
    auto after = gate.request(2 * ms, 0); CHECK(after);
    CHECK(after->generation != before->generation);
    CHECK(gate.accept(*after, 3 * ms, ms));
  }
  {
    auto gate = started();
    auto request = gate.request(ms, ms); CHECK(request);
    CHECK(gate.accept(*request, 2 * ms, 2 * ms));
    tick(gate, 76 * ms, 76 * ms, true, true);
    auto late = gate.request(76 * ms, 76 * ms); CHECK(late);
    CHECK(gate.accept(*late, 77 * ms, 77 * ms));
    CHECK(!gate.request(77 * ms, 77 * ms)); // No catch-up burst.
    tick(gate, 81 * ms, 81 * ms, true, true);
    CHECK(gate.request(81 * ms, 81 * ms)); // Original 20 ms phase.
    gate.fail(Fault::transport_closed); CHECK(!gate.enabled());
  }
  std::cout << "Native lifecycle checks passed\n";
}
