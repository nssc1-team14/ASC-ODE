#include <iostream>
#include <fstream>
#include <string>
#include <cmath>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <implicitRK.hpp>

using namespace ASC_ode;


class RCCircuit : public NonlinearFunction
{
private:
  double R, C, E0;
public:
  RCCircuit(double R_, double C_, double E0_) : R(R_), C(C_), E0(E0_) {}
  size_t dimX() const override { return 1; }
  size_t dimF() const override { return 1; }

  void evaluate(VectorView<double> x, VectorView<double> f) const override
  {
    double a = -1.0 / (R*C);
    double b =  E0  / (R*C);
    f(0) = a * x(0) + b;
  }

  void evaluateDeriv(VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    df(0,0) = -1.0 / (R*C);
  }
};


int main()
{
  double R = 1.0;
  double C = 1.0;
  double E0 = 1.0;
  double u0 = 0.0;

  auto rhs = std::make_shared<RCCircuit>(R, C, E0);

  int steps = 200;
  double tau = 10.0 / steps;

  Vector<> u = {u0};


  ExplicitEuler stepper(rhs);
  // ImplicitEuler stepper(rhs);
  // ImprovedEuler stepper(rhs);
  // CrankNicolson stepper(rhs);

  double t = 0.0;
  std::ofstream outfile ("output_test_rc.txt");
  outfile << t << "  " << u(0) << std::endl;

  for (int i = 0; i < steps; i++)
  {
    stepper.DoStep(tau, u);
    t += tau;
    outfile << t << "  " << u(0) << std::endl;
  }
  return 0;
};