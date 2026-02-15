#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#define TEST_EXPECT_TRUE(cond)                                                            \
  do {                                                                                     \
    if (!(cond)) {                                                                         \
      std::cerr << "Expectation failed at " << __FILE__ << ":" << __LINE__ << ": "  \
                << #cond << std::endl;                                                     \
      return false;                                                                        \
    }                                                                                      \
  } while (0)

#define TEST_EXPECT_NEAR(a, b, eps)                                                        \
  do {                                                                                     \
    const auto _a = (a);                                                                   \
    const auto _b = (b);                                                                   \
    if (std::fabs(static_cast<double>(_a) - static_cast<double>(_b)) >                    \
        static_cast<double>(eps)) {                                                        \
      std::cerr << "Expectation failed at " << __FILE__ << ":" << __LINE__             \
                << ": |" << #a << " - " << #b << "| > " << #eps                     \
                << " (" << _a << " vs " << _b << ")" << std::endl;                  \
      return false;                                                                        \
    }                                                                                      \
  } while (0)

inline int runTests(const std::vector<std::pair<std::string, std::function<bool()>>>& tests) {
  int passed = 0;
  for (const auto& [name, fn] : tests) {
    const bool ok = fn();
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << std::endl;
    if (ok) {
      ++passed;
    }
  }

  std::cout << "Passed " << passed << " / " << tests.size() << " tests." << std::endl;
  return passed == static_cast<int>(tests.size()) ? 0 : 1;
}

