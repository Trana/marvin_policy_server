#include "protocol.hpp"
#include <iomanip>
#include <iostream>
#include <limits>

using namespace marvin::native_io;
#define CHECK(condition) do { if (!(condition)) throw std::runtime_error(#condition); } while (false)

template<class F> void rejects(F fn) {
  bool rejected = false;
  try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
  CHECK(rejected);
}

int main(int argc, char **) {
  Packet p{Kind::snapshot, 1, 2, 3, 4, 5, 6, 7, 8, {0.0, -1.5, 9.81}};
  auto bytes = encode(p);
  if (argc == 2) {
    for (auto byte : bytes) std::cout << std::hex << std::setw(2) << std::setfill('0') << unsigned(byte);
    std::cout << '\n';
    return 0;
  }
  CHECK(bytes.size() == header_size + 24);
  auto decoded = decode(bytes.data(), bytes.size());
  CHECK(encode(decoded) == bytes);
  for (size_t size = 0; size < bytes.size(); ++size)
    rejects([&] { decode(bytes.data(), size); });
  for (size_t offset : {0U, 4U, 8U, 12U}) {
    auto bad = bytes; bad[offset] = 255;
    rejects([&] { decode(bad.data(), bad.size()); });
  }
  auto extra = bytes; extra.push_back(0);
  rejects([&] { decode(extra.data(), extra.size()); });
  for (double bad : {std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::quiet_NaN()}) {
    auto malformed = bytes;
    std::memcpy(malformed.data() + header_size, &bad, sizeof(bad));
    rejects([&] { decode(malformed.data(), malformed.size()); });
    p.values = {bad}; rejects([&] { encode(p); });
  }
  p.values.assign(max_values, 1.0);
  auto largest = encode(p); CHECK(largest.size() == max_packet - max_token_bytes);
  CHECK(encode(decode(largest.data(), largest.size())) == largest);
  p.values.push_back(1.0); rejects([&] { encode(p); });
  p.values.clear(); p.kind = static_cast<Kind>(0); rejects([&] { encode(p); });
  p.kind = Kind::prepare; p.token = "opaque-supervisor-id"; p.values.assign(12, .2);
  bytes = encode(p); CHECK(decode(bytes.data(), bytes.size()).token == p.token);
  p.token.assign(max_token_bytes + 1, 'a'); rejects([&] { encode(p); });
  std::cout << "Native protocol checks passed\n";
}
