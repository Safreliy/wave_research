// Reproduce Aphros PLIC flux at exactly zero face flux for a flat interface.
#include <cmath>
#include <iomanip>
#include <iostream>

#include "solver/reconst.h"

int main() {
  using R = Reconst<double>;
  R::Vect2 n2(0.0);
  R::Vect2 h2(0.125);
  n2[1] = 1.0;
  const double a2 = R::GetLineA(n2, 0.0005, h2);
  const double q2 = R::GetLineFlux(n2, a2, h2, 0.0, 0.00125, 1);
  R::Vect3 n3(0.0);
  R::Vect3 h3(0.125);
  n3[1] = 1.0;
  h3[2] = 1.0;
  const double a3 = R::GetLineA(n3, 0.0005, h3);
  const double q3 = R::GetLineFlux(n3, a3, h3, 0.0, 0.00125, 1);
  std::cout << std::setprecision(17)
            << "2D a=" << a2 << " zero_flux=" << q2
            << " finite=" << std::isfinite(q2) << '\n'
            << "3D a=" << a3 << " zero_flux=" << q3
            << " finite=" << std::isfinite(q3) << '\n';
}
