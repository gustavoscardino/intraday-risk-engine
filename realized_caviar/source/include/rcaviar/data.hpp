#pragma once
#include "rcaviar/model.hpp"
#include <cstdint>
#include <string>
#include <vector>
namespace rcaviar {
struct Price { std::int64_t timestamp; double price; };
struct Window { std::int64_t forecast_time, target_time; Observation observation; };
// Strict timestamp,price CSV. Integer seconds; contiguous 5-second samples.
std::vector<Price> read_prices(const std::string& path);
std::vector<Price> synthetic_prices(std::size_t samples = 120000, unsigned seed = 42);
// One window every minute, after 60 five-second returns; target is next 60.
std::vector<Window> make_windows(const std::vector<Price>& prices);
}
