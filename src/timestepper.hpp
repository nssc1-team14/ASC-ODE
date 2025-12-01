#ifndef TIMERSTEPPER_HPP
#define TIMERSTEPPER_HPP

#include <functional>
#include <exception>

#include "Newton.hpp"


namespace ASC_ode
{
  
  class TimeStepper
  { 
  protected:
    std::shared_ptr<NonlinearFunction> m_rhs;
  public:
    TimeStepper(std::shared_ptr<NonlinearFunction> rhs) : m_rhs(rhs) {}
    virtual ~TimeStepper() = default;
    virtual void DoStep(double tau, VectorView<double> y) = 0;
  };

  class ExplicitEuler : public TimeStepper
  {
    Vector<> m_vecf;
  public:
    ExplicitEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_vecf(rhs->dimF()) {}
    void DoStep(double tau, VectorView<double> y) override
    {
      this->m_rhs->evaluate(y, m_vecf);
      y += tau * m_vecf;
    }
  };

  class ImplicitEuler : public TimeStepper
  {
    std::shared_ptr<NonlinearFunction> m_equ;
    std::shared_ptr<Parameter> m_tau;
    std::shared_ptr<ConstantFunction> m_yold;
  public:
    ImplicitEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_tau(std::make_shared<Parameter>(0.0)) 
    {
      m_yold = std::make_shared<ConstantFunction>(rhs->dimX());
      auto ynew = std::make_shared<IdentityFunction>(rhs->dimX());
      m_equ = ynew - m_yold - m_tau * m_rhs;
    }

    void DoStep(double tau, VectorView<double> y) override
    {
      m_yold->set(y);
      m_tau->set(tau);
      NewtonSolver(m_equ, y);
    }
  };

  class ImprovedEuler : public TimeStepper
  {
    Vector<> m_vecf;
    Vector<> m_y_tilde;

  public:
    ImprovedEuler(std::shared_ptr<NonlinearFunction> rhs)
        : TimeStepper(rhs),
          m_vecf(rhs->dimF()),
          m_y_tilde(rhs->dimF())
    {}

    void DoStep(double tau, VectorView<double> y) override
    {
      // 1) f(y_n)
      this->m_rhs->evaluate(y, m_vecf);

      // 2) y_tilde = y_n + (tau/2) * f(y_n)
      for (std::size_t i = 0; i < y.size(); ++i)
        m_y_tilde[i] = y[i] + 0.5 * tau * m_vecf[i];

      // 3) f(y_tilde)
      this->m_rhs->evaluate(m_y_tilde, m_vecf);

      // 4) y_{n+1} = y_n + tau * f(y_tilde)
      y += tau * m_vecf;
    }
  };


  class CrankNicolson : public TimeStepper
  {
    std::shared_ptr<CrankNicolsonEquation> m_equ;

  public:
    CrankNicolson(std::shared_ptr<NonlinearFunction> rhs)
      : TimeStepper(rhs),
        m_equ(std::make_shared<CrankNicolsonEquation>(rhs))
    { }

    void DoStep(double tau, VectorView<double> y) override
    {
      // y is y_old on input; Newton overwrites it with y_new
      m_equ->set(tau, y);
      NewtonSolver(m_equ, y);
    }
  };

  class ExplicitRungeKutta : public TimeStepper
  {
    Matrix<> m_a;
    Vector<> m_b, m_c;
    int m_stages;
    int m_n;
    Vector<> m_k;
    Vector<> m_ytemp;

  public:
    ExplicitRungeKutta(std::shared_ptr<NonlinearFunction> rhs,
                       const Matrix<> &a,
                       const Vector<> &b,
                       const Vector<> &c)
      : TimeStepper(rhs),
        m_a(a), m_b(b), m_c(c),
        m_stages(c.size()),
        m_n(rhs->dimX()),
        m_k(m_stages * m_n),
        m_ytemp(m_n)
    { }

    void DoStep(double tau, VectorView<double> y) override
    {

      for (int j = 0; j < m_stages; ++j)
      {
        m_ytemp = y;

        for (int l = 0; l < j; ++l)
        {
          auto kl = m_k.range(l * m_n, (l + 1) * m_n);
          m_ytemp += tau * m_a(j, l) * kl;
        }

        auto kj = m_k.range(j * m_n, (j + 1) * m_n);
        this->m_rhs->evaluate(m_ytemp, kj);
      }

      for (int j = 0; j < m_stages; ++j)
      {
        auto kj = m_k.range(j * m_n, (j + 1) * m_n);
        y += tau * m_b(j) * kj;
      }
    }
  };

}


#endif
