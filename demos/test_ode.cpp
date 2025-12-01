#include <iostream>
#include <fstream> 
#include <string>
#include <cmath>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>
#include <implicitRK.hpp>

using namespace ASC_ode;


class MassSpring : public NonlinearFunction
{
private:
  double mass;
  double stiffness;

public:
  MassSpring(double m, double k) : mass(m), stiffness(k) {}

  size_t dimX() const override { return 2; }
  size_t dimF() const override { return 2; }
  
  void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f(0) = x(1);
    f(1) = -stiffness/mass*x(0);
  }
  
  void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    df = 0.0;
    df(0,1) = 1;
    df(1,0) = -stiffness/mass;
  }
};


int main()
{
  double tend = 20*M_PI;
  int steps = 1000;
  double tau = tend/steps;

  Vector<> y = { 1, 0 };  // initializer list
  auto rhs = std::make_shared<MassSpring>(1.0, 1.0);


/*
  Vector<> Radau(3), RadauWeight(3);
  GaussRadau (Radau, RadauWeight);
  // not sure about weights, comput them via ComputeABfromC
  cout << "Radau = " << Radau << ", weight = " << RadauWeight <<  endl;
        Vector<> Gauss2c(2), Gauss3c(3);
*/


  // ExplicitEuler stepper(rhs);
  // ImplicitEuler stepper(rhs);
  ImprovedEuler stepper(rhs);

  // RungeKutta stepper(rhs, Gauss2a, Gauss2b, Gauss2c);

  // Gauss3c .. points tabulated, compute a,b:
  // auto [Gauss3a,Gauss3b] = ComputeABfromC (Gauss3c);
  // ImplicitRungeKutta stepper(rhs, Gauss3a, Gauss3b, Gauss3c);


  /*
  // arbitrary order Gauss-Legendre
  int stages = 5;
  Vector<> c(stages), b1(stages);
  GaussLegendre(c, b1);

  auto [a, b] = ComputeABfromC(c);
  ImplicitRungeKutta stepper(rhs, a, b, c);
  */

  /*
  // arbitrary order Radau
  int stages = 5;
  Vector<> c(stages), b1(stages);
  GaussRadau(c, b1);

  auto [a, b] = ComputeABfromC(c);
  ImplicitRungeKutta stepper(rhs, a, b, c);
  */


  std::ofstream outfile ("output_test_ode.txt");
  std::cout << 0.0 << "  " << y(0) << " " << y(1) << std::endl;
  outfile << 0.0 << "  " << y(0) << " " << y(1) << std::endl;

  for (int i = 0; i < steps; i++)
  {
     stepper.DoStep(tau, y);

     std::cout << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
     outfile << (i+1) * tau << "  " << y(0) << " " << y(1) << std::endl;
  }

  // Explicit Euler
  {
    Vector<> y_exp = {1, 0};
    ExplicitEuler stepper_exp(rhs);
    std::ofstream out_exp("massspring_explicit.txt");

    double t = 0.0;
    out_exp << t << " " << y_exp(0) << " " << y_exp(1) << "\n";
    for (int i = 0; i < steps; ++i)
    {
      stepper_exp.DoStep(tau, y_exp);
      t += tau;
      out_exp << t << " " << y_exp(0) << " " << y_exp(1) << "\n";
    }
  }

  // Implicit Euler
  {
    Vector<> y_imp = {1, 0};
    ImplicitEuler stepper_imp(rhs);
    std::ofstream out_imp("massspring_implicit.txt");

    double t = 0.0;
    out_imp << t << " " << y_imp(0) << " " << y_imp(1) << "\n";
    for (int i = 0; i < steps; ++i)
    {
      stepper_imp.DoStep(tau, y_imp);
      t += tau;
      out_imp << t << " " << y_imp(0) << " " << y_imp(1) << "\n";
    }
  }

  // Crank–Nicolson
  {
    Vector<> y_cn = {1, 0};
    CrankNicolson stepper_cn(rhs);
    std::ofstream out_cn("massspring_cn.txt");

    double t = 0.0;
    out_cn << t << " " << y_cn(0) << " " << y_cn(1) << "\n";
    for (int i = 0; i < steps; ++i)
    {
      stepper_cn.DoStep(tau, y_cn);
      t += tau;
      out_cn << t << " " << y_cn(0) << " " << y_cn(1) << "\n";
    }
  }


  std::cout << "\n\n=== RC CIRCUIT SIMULATION ===\n";

  // RC parameters
  double R = 1.0;
  double C = 1.0;
  double E0 = 1.0;
  double u0 = 0.0; // initial capacitor voltage

  // Create RHS NonlinearFunction
  class RCCircuit : public NonlinearFunction
  {
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

  auto rc_rhs = std::make_shared<RCCircuit>(R, C, E0);

  int rc_steps = 200;
  double rc_tau = 10.0 / rc_steps;

  // ------------------ Explicit Euler ------------------
  {
    Vector<> u = {u0};
    ExplicitEuler stepper_rc(rc_rhs);
    std::ofstream out("rc_explicit.txt");

    double t = 0.0;
    out << t << " " << u(0) << "\n";
    for (int i = 0; i < rc_steps; i++)
    {
      stepper_rc.DoStep(rc_tau, u);
      t += rc_tau;
      out << t << " " << u(0) << "\n";
    }
  }

  // ------------------ Implicit Euler ------------------
  {
    Vector<> u = {u0};
    ImplicitEuler stepper_rc(rc_rhs);
    std::ofstream out("rc_implicit.txt");

    double t = 0.0;
    out << t << " " << u(0) << "\n";
    for (int i = 0; i < rc_steps; i++)
    {
      stepper_rc.DoStep(rc_tau, u);
      t += rc_tau;
      out << t << " " << u(0) << "\n";
    }
  }

  // ------------------ Crank–Nicolson ------------------
  {
    Vector<> u = {u0};
    CrankNicolson stepper_rc(rc_rhs);
    std::ofstream out("rc_cn.txt");

    double t = 0.0;
    out << t << " " << u(0) << "\n";
    for (int i = 0; i < rc_steps; i++)
    {
      stepper_rc.DoStep(rc_tau, u);
      t += rc_tau;
      out << t << " " << u(0) << "\n";
    }
  }
  return 0;
}
