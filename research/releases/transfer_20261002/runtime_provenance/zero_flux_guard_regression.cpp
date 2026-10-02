// Exercise exact zero PLIC transport and bitwise nonzero-output preservation.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

#include "solver/reconst.h"

static std::uint64_t bits(double x) {
  std::uint64_t result;
  static_assert(sizeof(result) == sizeof(x), "unexpected double width");
  std::memcpy(&result, &x, sizeof(result));
  return result;
}

template <class Vect>
static void trial(const char* dim, const char* normal_name, Vect n, Vect h) {
  using R = Reconst<double>;
  const std::size_t d = 1;
  const double dt = 0.00125;
  for (double u : {0.0, 0.0005, 0.5, 1.0}) {
    const double a = R::GetLineA(n, u, h);
    const double v0 = R::GetLineVol(n, a, h, 0.0, d);
    const double vs0 = R::GetLineVolStr(n, a, h, 0.0, 0.001, d);
    const double f0 = R::GetLineFlux(n, a, h, 0.0, dt, d);
    const double fs0 = R::GetLineFluxStr(n, a, h, 0.0, 0.001, dt, d);
    const bool okay = std::isfinite(v0) && v0 == 0.0 &&
                      std::isfinite(vs0) && vs0 == 0.0 &&
                      std::isfinite(f0) && f0 == 0.0 &&
                      std::isfinite(fs0) && fs0 == 0.0;
    std::cout << "zero " << dim << ' ' << normal_name << ' ' << u
              << " vol=" << v0 << " volstr=" << vs0
              << " flux=" << f0 << " fluxstr=" << fs0
              << " pass=" << okay << '\n';
    if (u > 0.0 && u < 1.0) {
      for (double q : {-0.0001, 0.0001}) {
        const double f = R::GetLineFlux(n, a, h, q, dt, d);
        const double fs = R::GetLineFluxStr(n, a, h, q, q * 0.8, dt, d);
        std::cout << "nonzero " << dim << ' ' << normal_name << ' ' << u
                  << ' ' << q << " fluxbits=" << std::hex << bits(f)
                  << " fluxstrbits=" << bits(fs) << std::dec << '\n';
      }
    }
  }
}

int main() {
  using R = Reconst<double>;
  R::Vect2 h2(0.125), axis2(0.0), oblique2(0.0);
  axis2[1] = 1.0;
  oblique2[0] = 0.6;
  oblique2[1] = 0.8;
  trial("2D", "axis", axis2, h2);
  trial("2D", "oblique", oblique2, h2);
  R::Vect3 h3(0.125), axis3(0.0), oblique3(0.0);
  h3[2] = 1.0;
  axis3[1] = 1.0;
  oblique3[0] = 0.6;
  oblique3[1] = 0.8;
  trial("3D", "axis", axis3, h3);
  trial("3D", "oblique", oblique3, h3);
}
