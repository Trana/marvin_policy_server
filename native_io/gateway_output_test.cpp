#include "gateway_output.hpp"
#include <iostream>
#include <limits>
using namespace marvin::native_io;
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (false)
int main() {
  GatewayOutput output;
  Packet p; p.kind = Kind::prepare; p.token = "supervisor-generation-1";
  p.values.assign(12, .2); p.requested_ns = 5000000100ULL;
  CHECK(!output.ready(100));
  CHECK(!output.prepare(p, 100, false));
  CHECK(!output.prepare(p, 99, true)); // expiry cannot be extended by transport
  CHECK(output.prepare(p, 100, true));
  CHECK(output.ready(100)); CHECK(!output.ready(p.requested_ns+1));
  output.start(7, 1000000000);
  CHECK(!output.ready(100)); CHECK(output.owns(7)); CHECK(!output.owns(6));
  Joints targets{}; targets.fill(.6);
  CHECK(output.command(targets, 7, 1000000000)[0] == .2);
  CHECK(std::abs(output.command(targets, 7, 2000000000)[0] - .4) < 1e-12);
  CHECK(output.command(targets, 7, 3000000000)[0] == .6);
  CHECK(output.sequence() == 3); CHECK(output.token() == p.token);
  output.rebind(8); CHECK(!output.owns(7)); CHECK(output.owns(8));
  output.stop(); CHECK(output.released()); CHECK(!output.owns(8)); CHECK(!output.ready(100));
  bool rejected = false;
  try { output.command(targets, 8, 4000000000); } catch (const std::logic_error &) { rejected = true; }
  CHECK(rejected);
  p.values[0] = std::numeric_limits<double>::quiet_NaN(); CHECK(!output.prepare(p, 100, true));
  p.values[0] = 0.; p.token.clear(); CHECK(!output.prepare(p, 100, true));
  std::cout << "Gateway preparation, blend, epoch and release checks passed\n";
}
