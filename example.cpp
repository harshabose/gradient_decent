#include <iostream>
#include <chrono>

#include "gradient_decent.h"

double bivarient_function (const double x, const double y) noexcept {
  constexpr double A = 10;
  return (A * x * y) / (std::exp(x * x + y * y)) + (5.0/std::exp(1.0));
}

int main () {
  std::tuple<double, double> lower_bounds = {-2.0F, -2.0F};
  std::tuple<double, double> upper_bounds = {2.0F, 2.0F};

  std::unique_ptr<gd::gradient_decent<double, double, double>> gradient_operator;
  gradient_operator = std::make_unique<gd::gradient_decent<double, double, double>>(bivarient_function, 1.6, -1.2);

  gradient_operator->add_lower_bounds(lower_bounds);
  gradient_operator->add_upper_bounds(upper_bounds);
  gradient_operator->set_tolerance(1e-3);
//        gradient_operator->toggle_classic_gradient_algo();  // uncomment this to use classic GD

  auto start = std::chrono::high_resolution_clock::now();
  gradient_operator->perform_gradient_decent();
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;
}
