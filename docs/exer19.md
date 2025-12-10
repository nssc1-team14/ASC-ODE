
# Exercise 19 — Runge–Kutta Comparison on the Mass–Spring ODE

## Explicit Runge–Kutta methods

### Explicit RK4 (classic, 4-stage)
- Tracks the exact oscillation closely; phase plot is almost perfectly elliptical.

Time | Phase
:---:|:----:
![Explicit RK4 time](pictures/explicitrk4_time.svg) | ![Explicit RK4 phase](pictures/explicitrk4_phase.svg)

## Implicit Runge–Kutta (Gauss/Radau)

### Gauss–Legendre 2-stage
- Symplectic and A-stable; good energy behavior with tight ellipse.

Time | Phase
:---:|:----:
![Gauss–Legendre 2-stage time](pictures/gausslegendre2rk_time.svg) | ![Gauss–Legendre 2-stage phase](pictures/gausslegendre2rk_phase.svg)

### Gauss–Legendre 3-stage
- Higher order reduces phase error further; trajectory nearly overlays the reference.

Time | Phase
:---:|:----:
![Gauss–Legendre 3-stage time](pictures/gausslegendre3rk_time.svg) | ![Gauss–Legendre 3-stage phase](pictures/gausslegendre3rk_phase.svg)

### Arbitrary-order Gauss–Legendre
- Very high accuracy over the 4π window; ellipse visually matches the analytical solution.

Time | Phase
:---:|:----:
![Arbitrary-order Gauss–Legendre time](pictures/ao_gausslegendrerk_time.svg) | ![Arbitrary-order Gauss–Legendre phase](pictures/ao_gausslegendrerk_phase.svg)

### Arbitrary-order Radau IIA
- L-stable; noticeable damping as expected for stiff-friendly schemes.

Time | Phase
:---:|:----:
![Radau IIA time](pictures/ao_radaurk_time.svg) | ![Radau IIA phase](pictures/ao_radaurk_phase.svg)

