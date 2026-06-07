//! Classical fourth-order Runge-Kutta method (RK4).
//!
//! Uses the Butcher tableau:
//! ```text
//! k1 = f(x_n,       y_n)
//! k2 = f(x_n + h/2, y_n + h·k1/2)
//! k3 = f(x_n + h/2, y_n + h·k2/2)
//! k4 = f(x_n + h,   y_n + h·k3)
//! y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
//! ```
//!
//! Global truncation error: O(h⁴).

/// Solve an IVP using the classical RK4 method.
///
/// # Arguments
///
/// * `f`     — Right-hand side `dy/dx = f(x, y)`.
/// * `x0`    — Initial independent variable.
/// * `y0`    — Initial value.
/// * `x_end` — Terminal value.
/// * `n`     — Number of steps (≥ 1).
///
/// # Returns
///
/// `(xs, ys)` — vectors of `x` and `y` at each step.
pub fn solve(f: &dyn Fn(f64, f64) -> f64, x0: f64, y0: f64, x_end: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 1, "number of steps must be ≥ 1");
    let h = (x_end - x0) / n as f64;
    let mut xs = Vec::with_capacity(n + 1);
    let mut ys = Vec::with_capacity(n + 1);
    xs.push(x0);
    ys.push(y0);
    let mut x = x0;
    let mut y = y0;
    for _ in 0..n {
        let k1 = f(x, y);
        let k2 = f(x + h / 2.0, y + h * k1 / 2.0);
        let k3 = f(x + h / 2.0, y + h * k2 / 2.0);
        let k4 = f(x + h, y + h * k3);
        y += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        x += h;
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_growth() {
        let f = |_x: f64, y: f64| y;
        let (_, ys) = solve(&f, 0.0, 1.0, 1.0, 10);
        let exact = 1.0_f64.exp();
        assert!((ys[10] - exact).abs() < 1e-5, "got {}, exact {}", ys[10], exact);
    }

    #[test]
    fn exponential_growth_high_accuracy() {
        let f = |_x: f64, y: f64| y;
        let (_, ys) = solve(&f, 0.0, 1.0, 1.0, 100);
        let exact = 1.0_f64.exp();
        assert!((ys[100] - exact).abs() < 1e-8);
    }

    /// RK4 is fourth-order: error ∝ h⁴
    #[test]
    fn convergence_order_rk4() {
        let f = |_x: f64, y: f64| y;
        let exact = 1.0_f64.exp();
        let e1 = (solve(&f, 0.0, 1.0, 1.0, 10).1[10] - exact).abs();
        let e2 = (solve(&f, 0.0, 1.0, 1.0, 20).1[20] - exact).abs();
        let ratio = e1 / e2;
        // ratio should be ~16 (2⁴) for fourth-order method
        assert!(ratio > 10.0 && ratio < 25.0, "convergence ratio = {ratio}, expected ~16");
    }

    #[test]
    fn constant_rhs() {
        let f = |_x, _y| 0.0;
        let (_, ys) = solve(&f, 0.0, 42.0, 1.0, 10);
        for &y in &ys {
            assert!((y - 42.0).abs() < 1e-12);
        }
    }

    #[test]
    fn linear_rhs() {
        let f = |_x, _y| 2.0;
        let (_, ys) = solve(&f, 0.0, 0.0, 3.0, 100);
        // exact: y = 2x, y(3) = 6
        assert!((ys[100] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn quadratic_rhs() {
        // dy/dx = 2x → y = x²
        let f = |x: f64, _y: f64| 2.0 * x;
        let (_, ys) = solve(&f, 0.0, 0.0, 5.0, 100);
        assert!((ys[100] - 25.0).abs() < 1e-8);
    }

    #[test]
    fn sinusoidal_rhs() {
        // dy/dx = cos(x) → y = sin(x)
        let f = |x: f64, _y: f64| x.cos();
        let (_, ys) = solve(&f, 0.0, 0.0, std::f64::consts::PI / 2.0, 100);
        assert!((ys[100] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn exponential_decay() {
        let f = |_x: f64, y: f64| -y;
        let (_, ys) = solve(&f, 0.0, 1.0, 2.0, 100);
        assert!((ys[100] - (-2.0_f64).exp()).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "number of steps must be ≥ 1")]
    fn panics_on_zero_steps() {
        let f = |_x, _y| 0.0;
        solve(&f, 0.0, 1.0, 1.0, 0);
    }

    #[test]
    fn single_step() {
        let f = |_x, y| y;
        let (xs, ys) = solve(&f, 0.0, 1.0, 1.0, 1);
        assert_eq!(xs.len(), 2);
        assert_eq!(ys.len(), 2);
        let k1 = 1.0_f64;
        let k2 = 1.5_f64; // f(0.5, 1.5) = 1.5
        let k3 = 1.75_f64; // f(0.5, 1.75) = 1.75
        let k4 = 2.75_f64; // f(1.0, 2.75) = 2.75
        let expected = 1.0 + (1.0 / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        assert!((ys[1] - expected).abs() < 1e-12);
    }

    #[test]
    fn negative_direction() {
        let f = |_x, y| y;
        let (_, ys) = solve(&f, 1.0, 1.0_f64.exp(), 0.0, 100);
        assert!((ys[100] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn stiff_problem_moderate() {
        // dy/dx = -15y, moderate stiffness
        let f = |_x, y| -15.0 * y;
        let (_, ys) = solve(&f, 0.0, 1.0, 1.0, 10_000);
        let exact = (-15.0_f64).exp();
        assert!((ys[10_000] - exact).abs() < 1e-3);
    }
}
