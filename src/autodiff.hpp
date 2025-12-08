#ifndef AUTODIFF_HPP
#define AUTODIFF_HPP

#include <cstddef> 
#include <ostream> 
#include <cmath>   
#include <array>  
#include <vector>


namespace ASC_ode
{

  template <size_t N, typename T = double>
  class Variable 
  {
    private:
      T m_val;
    public:
      Variable (T v) : m_val(v) {}
      T value() const { return m_val; }
  };

  template <typename T = double>
  auto derivative (T v, size_t /*index*/) { return T(0); } 


  template <size_t N, typename T = double>
  class AutoDiff
  {
  private:
    T m_val;
    std::array<T, N> m_deriv;
  public: 
    AutoDiff () : m_val(0), m_deriv{} {}
    AutoDiff (T v) : m_val(v), m_deriv{} 
    {
      for (size_t i = 0; i < N; i++)
        m_deriv[i] = derivative(v, i);
    }
    
    template <size_t I>
    AutoDiff (Variable<I, T> var) : m_val(var.value()), m_deriv{} 
    {
      m_deriv[I] = 1.0;
    }

    T value() const { return m_val; }
    std::array<T, N>& deriv() { return m_deriv; }
    const std::array<T, N>& deriv() const { return m_deriv; }
  };


  template <size_t N, typename T = double>
  auto derivative (AutoDiff<N, T> v, size_t index) 
  {
    return v.deriv()[index];
  }



  template <size_t N, typename T>
  std::ostream & operator<< (std::ostream& os, const AutoDiff<N, T>& ad)
  {
    os << "Value: " << ad.value() << ", Deriv: [";
    for (size_t i = 0; i < N; i++)
    {
      os << ad.deriv()[i];
      if (i < N - 1) os << ", ";
    }
    os << "]";
    return os;
  }

  template <size_t N, typename T = double>
  AutoDiff<N, T> operator+ (const AutoDiff<N, T>& a, const AutoDiff<N, T>& b)
  {
     AutoDiff<N, T> result(a.value() + b.value());
     for (size_t i = 0; i < N; i++)
        result.deriv()[i] = a.deriv()[i] + b.deriv()[i];
       return result;
   }

  // scalar + AD
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator+ (T a, const AutoDiff<N, T>& b)
  {
    AutoDiff<N, T> result(a + b.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = b.deriv()[i];
    return result;
  }

  // AD + scalar
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator+ (const AutoDiff<N, T>& a, T b)
  {
    AutoDiff<N, T> result(a.value() + b);
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a.deriv()[i];
    return result;
  }

  // AD - AD
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator- (const AutoDiff<N, T>& a, const AutoDiff<N, T>& b)
  {
    AutoDiff<N, T> result(a.value() - b.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a.deriv()[i] - b.deriv()[i];
    return result;
  }

  // scalar - AD
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator- (T a, const AutoDiff<N, T>& b)
  {
    AutoDiff<N, T> result(a - b.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = -b.deriv()[i];
    return result;
  }

  // AD - scalar
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator- (const AutoDiff<N, T>& a, T b)
  {
    AutoDiff<N, T> result(a.value() - b);
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a.deriv()[i];
    return result;
  }

  // unary minus
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator- (const AutoDiff<N, T>& a)
  {
    AutoDiff<N, T> result(-a.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = -a.deriv()[i];
    return result;
  }

   template <size_t N, typename T = double>
   auto operator+ (T a, const AutoDiff<N, T>& b) { return AutoDiff<N, T>(a) + b; }


   template <size_t N, typename T = double>
   AutoDiff<N, T> operator* (const AutoDiff<N, T>& a, const AutoDiff<N, T>& b)
   {
       AutoDiff<N, T> result(a.value() * b.value());
       for (size_t i = 0; i < N; i++)
          result.deriv()[i] = a.deriv()[i] * b.value() + a.value() * b.deriv()[i];
       return result;
   }

   // scalar * AD
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator* (T a, const AutoDiff<N, T>& b)
  {
    AutoDiff<N, T> result(a * b.value());
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a * b.deriv()[i];
    return result;
  }

  // AD * scalar
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator* (const AutoDiff<N, T>& a, T b)
  {
    return b * a;
  }

  // AD / AD
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator/ (const AutoDiff<N, T>& a, const AutoDiff<N, T>& b)
  {
    AutoDiff<N, T> result(a.value() / b.value());
    T denom = b.value() * b.value();
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] =
        (a.deriv()[i] * b.value() - a.value() * b.deriv()[i]) / denom;
    return result;
  }

  // AD / scalar
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator/ (const AutoDiff<N, T>& a, T b)
  {
    AutoDiff<N, T> result(a.value() / b);
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a.deriv()[i] / b;
    return result;
  }

  // scalar / AD
  template <size_t N, typename T = double>
  AutoDiff<N, T> operator/ (T a, const AutoDiff<N, T>& b)
  {
    AutoDiff<N, T> result(a / b.value());
    T denom = b.value() * b.value();
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = -a * b.deriv()[i] / denom;
    return result;
  }


   using std::sin;
   using std::cos;
   using std::exp;
   using std::log;

   template <size_t N, typename T = double>
   AutoDiff<N, T> sin(const AutoDiff<N, T> &a)
   {
       AutoDiff<N, T> result(sin(a.value()));
       for (size_t i = 0; i < N; i++)
           result.deriv()[i] = cos(a.value()) * a.deriv()[i];
       return result;
   }

   template <size_t N, typename T = double>
  AutoDiff<N, T> cos(const AutoDiff<N, T> &a)
  {
    AutoDiff<N, T> result(cos(a.value()));
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = -sin(a.value()) * a.deriv()[i];
    return result;
  }

  template <size_t N, typename T = double>
  AutoDiff<N, T> exp(const AutoDiff<N, T> &a)
  {
    T e = exp(a.value());
    AutoDiff<N, T> result(e);
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = e * a.deriv()[i];
    return result;
  }

  template <size_t N, typename T = double>
  AutoDiff<N, T> log(const AutoDiff<N, T> &a)
  {
    AutoDiff<N, T> result(log(a.value()));
    for (size_t i = 0; i < N; i++)
      result.deriv()[i] = a.deriv()[i] / a.value();
    return result;
  }

  // Legendre Polynomials (Exercise 18.4)
  template <typename T>
  void LegendrePolynomials(int n, T x, std::vector<T>& P)
  {
      if (n < 0) {
          P.clear();
          return;
      }
      P.resize(n + 1);
      P[0] = T(1);
      if (n == 0) return;

      P[1] = x;
      for (int k = 2; k <= n; ++k) {
          P[k] = ( (T(2 * k - 1) * x * P[k - 1])
                 - T(k - 1) * P[k - 2]) / T(k);
      }
  }

} // namespace ASC_ode


#endif
