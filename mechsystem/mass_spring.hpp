#ifndef MASS_SPRING_HPP
#define MASS_SPRING_HPP

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

using namespace ASC_ode;

#include <vector.hpp>
using namespace nanoblas;


template <int D>
class Mass
{
public:
  double mass;
  Vec<D> pos;
  Vec<D> vel = 0.0;
  Vec<D> acc = 0.0;
};


template <int D>
class Fix
{
public:
  Vec<D> pos;
};


class Connector
{
public:
  enum CONTYPE { FIX=1, MASS=2 };
  CONTYPE type;
  size_t nr;
};

std::ostream & operator<< (std::ostream & ost, const Connector & con)
{
  ost << "type = " << int(con.type) << ", nr = " << con.nr;
  return ost;
}

class Spring
{
public:
  double length;  
  double stiffness;
  std::array<Connector,2> connectors;
};

template <int D>
class MassSpringSystem
{
  std::vector<Fix<D>> m_fixes;
  std::vector<Mass<D>> m_masses;
  std::vector<Spring> m_springs;
  Vec<D> m_gravity=0.0;
public:
  void setGravity (Vec<D> gravity) { m_gravity = gravity; }
  Vec<D> getGravity() const { return m_gravity; }

  Connector addFix (Fix<D> p)
  {
    m_fixes.push_back(p);
    return { Connector::FIX, m_fixes.size()-1 };
  }

  Connector addMass (Mass<D> m)
  {
    m_masses.push_back (m);
    return { Connector::MASS, m_masses.size()-1 };
  }
  
  size_t addSpring (Spring s) 
  {
    m_springs.push_back (s); 
    return m_springs.size()-1;
  }

  auto & fixes() { return m_fixes; } 
  auto & masses() { return m_masses; } 
  auto & springs() { return m_springs; }

  void getState (VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    auto valmat = values.asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.asMatrix(m_masses.size(), D);

    for (size_t i = 0; i < m_masses.size(); i++)
      {
        valmat.row(i) = m_masses[i].pos;
        dvalmat.row(i) = m_masses[i].vel;
        ddvalmat.row(i) = m_masses[i].acc;
      }
  }

  void setState (VectorView<> values, VectorView<> dvalues, VectorView<> ddvalues)
  {
    auto valmat = values.asMatrix(m_masses.size(), D);
    auto dvalmat = dvalues.asMatrix(m_masses.size(), D);
    auto ddvalmat = ddvalues.asMatrix(m_masses.size(), D);

    for (size_t i = 0; i < m_masses.size(); i++)
      {
        m_masses[i].pos = valmat.row(i);
        m_masses[i].vel = dvalmat.row(i);
        m_masses[i].acc = ddvalmat.row(i);
      }
  }
};

template <int D>
std::ostream & operator<< (std::ostream & ost, MassSpringSystem<D> & mss)
{
  ost << "fixes:" << std::endl;
  for (auto f : mss.fixes())
    ost << f.pos << std::endl;

  ost << "masses: " << std::endl;
  for (auto m : mss.masses())
    ost << "m = " << m.mass << ", pos = " << m.pos << std::endl;

  ost << "springs: " << std::endl;
  for (auto sp : mss.springs())
    ost << "length = " << sp.length << ", stiffness = " << sp.stiffness
        << ", C1 = " << sp.connectors[0] << ", C2 = " << sp.connectors[1] << std::endl;
  return ost;
}


template <int D>
class MSS_Function : public NonlinearFunction
{
  MassSpringSystem<D> & mss;
public:
  MSS_Function (MassSpringSystem<D> & _mss)
    : mss(_mss) { }

  virtual size_t dimX() const override { return D*mss.masses().size(); }
  virtual size_t dimF() const override{ return D*mss.masses().size(); }

  virtual void evaluate (VectorView<double> x, VectorView<double> f) const override
  {
    f = 0.0;

    auto xmat = x.asMatrix(mss.masses().size(), D);
    auto fmat = f.asMatrix(mss.masses().size(), D);

    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) = mss.masses()[i].mass*mss.getGravity();

    for (auto spring : mss.springs())
      {
        auto [c1,c2] = spring.connectors;
        Vec<D> p1, p2;
        if (c1.type == Connector::FIX)
          p1 = mss.fixes()[c1.nr].pos;
        else
          p1 = xmat.row(c1.nr);
        if (c2.type == Connector::FIX)
          p2 = mss.fixes()[c2.nr].pos;
        else
          p2 = xmat.row(c2.nr);

        double force = spring.stiffness * (norm(p1-p2)-spring.length);
        Vec<D> dir12 = 1.0/norm(p1-p2) * (p2-p1);
        if (c1.type == Connector::MASS)
          fmat.row(c1.nr) += force*dir12;
        if (c2.type == Connector::MASS)
          fmat.row(c2.nr) -= force*dir12;
      }

    for (size_t i = 0; i < mss.masses().size(); i++)
      fmat.row(i) *= 1.0/mss.masses()[i].mass;
  }
  
  virtual void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
  {
    const size_t N   = mss.masses().size();
    const int    dim = D;
    const size_t nX  = dim * N;

    df = 0.0;

    // positions of masses from state vector
    auto xmat = x.asMatrix(N, dim);
    auto & masses  = mss.masses();
    auto & springs = mss.springs();
    auto & fixes   = mss.fixes();

    for (const auto & spring : springs)
    {
      const Connector &c1 = spring.connectors[0];
      const Connector &c2 = spring.connectors[1];

      // positions of the two connectors
      Vec<D> p1, p2;
      if (c1.type == Connector::FIX)
        p1 = fixes[c1.nr].pos;
      else
        p1 = xmat.row(c1.nr);

      if (c2.type == Connector::FIX)
        p2 = fixes[c2.nr].pos;
      else
        p2 = xmat.row(c2.nr);

      // d = p2 - p1, r = |d|
      Vec<D> d = p2 - p1;
      double r2 = 0.0;
      for (int k = 0; k < dim; ++k)
        r2 += d(k)*d(k);

      double r = std::sqrt(r2);
      if (r < 1e-12)   // avoid division by ~0
        continue;

      double L     = spring.length;
      double kappa = spring.stiffness;

      // F1 = kappa * (1 - L/r) * d
      // A = dF1/dd = kappa*(beta*I + (L/r^3)*d d^T)
      double beta   = 1.0 - L/r;
      double coeff1 = kappa * beta;
      double coeff2 = kappa * L / (r2 * r);   // = kappa*L / r^3

      Matrix<> A(dim, dim);
      for (int a = 0; a < dim; ++a)
        for (int b = 0; b < dim; ++b)
        {
          double val = coeff2 * d(a) * d(b);
          if (a == b)
            val += coeff1;
          A(a,b) = val;
        }

      // helper: add scaled A block at (rowMass, colMass)
      auto add_block = [&](int rowMass, int colMass, double scale)
      {
        int row0 = rowMass * dim;
        int col0 = colMass * dim;
        for (int a = 0; a < dim; ++a)
          for (int b = 0; b < dim; ++b)
            df(row0 + a, col0 + b) += scale * A(a,b);
      };

      // contributions for mass at c1 (force F1)
      if (c1.type == Connector::MASS)
      {
        int i      = static_cast<int>(c1.nr);
        double mi  = masses[i].mass;
        double s_i = 1.0 / mi;     // a_i = F_i / m_i

        // d a_i / d p1 = (1/mi) * dF1/dp1 = (-A)/mi   (p1 belongs to mass i)
        add_block(i, i, -s_i);

        // if c2 is mass j: d a_i / d p2 = (1/mi) * dF1/dp2 = (+A)/mi
        if (c2.type == Connector::MASS)
        {
          int j = static_cast<int>(c2.nr);
          add_block(i, j, +s_i);
        }
        // if c2 is FIX: no derivative wrt fixed position
      }

      // contributions for mass at c2 (force F2 = -F1)
      if (c2.type == Connector::MASS)
      {
        int j      = static_cast<int>(c2.nr);
        double mj  = masses[j].mass;
        double s_j = 1.0 / mj;     // a_j = F_j / m_j

        // d a_j / d p2 = (1/mj) * dF2/dp2 = (-A)/mj  (since dF2/dp2 = -A)
        add_block(j, j, -s_j);

        // if c1 is mass i: d a_j / d p1 = (1/mj) * dF2/dp1 = (+A)/mj
        if (c1.type == Connector::MASS)
        {
          int i = static_cast<int>(c1.nr);
          add_block(j, i, +s_j);
        }
      }
    }// gravity has no x-dependence → no contribution
  }
};

#endif
