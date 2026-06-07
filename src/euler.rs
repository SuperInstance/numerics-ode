//! Euler method (first-order explicit).
//!
//! Solves the initial value problem
//! ```text
//! dy/dx = f(x, y),  y(x₀) = y₀
//! ```
//! using the forward Euler scheme:
//! ```text
//! y_{n+1} = y_n + h · f(x_n, y_n)
//! ```
//!
//! Global truncation error: O(h).

/// Solve an IVP using the forward Euler method.
///
/// # Arguments
///
/// * `f`    — Right-hand side `dy/dx = f(x, y)`.
/// * `x0`   — Initial independent variable.
/// * `y0`   — Initial value.
/// * `x_end`— Terminal value of the independent variable.
/// * `n`    — Number of steps (≥ 1).
///
/// # Returns
///
/// `(xs, ys)` — vectors of `x` and `y` values at each step.
///
/// # Panics
///
/// Panics if `n` is zero.
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
        y = y + h * f(x, y);
        x += h;
        xs.push(x);
        ys.push(y);
    }
    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// dy/dx = 0  →  y = y0 (constant)
    #[test]
    fn constant_rhs() {
        let f = |_x: f64, _y: f64| 0.0_f64;
        let (_xs, ys) = solve(&f, 0.0, 5.0, 1.0, 10);
        assert_eq!(ys.len(), 11);
        for &y in &ys {
            assert!((y - 5.0).abs() < 1e-12, "y = {y}, expected 5.0");
        }
    }

    /// dy/dx = 1  →  y = x + y0
    #[test]
    fn linear_rhs() {
        let f = |_x: f64, _y: f64| 1.0;
        let (_, ys) = solve(&f, 0.0, 0.0, 2.0, 200);
        assert!((ys[200] - 2.0).abs() < 1e-4);
    }

    /// dy/dx = y  →  y = e^x
    #[test]
    fn exponential_growth() {
        let f = |_x: f64, y: f64| y;
        let (_, ys) = solve(&f, 0.0, 1.0, 1.0, 10_000);
        let exact = 1.0_f64.exp();
        assert!((ys[10_000] - exact).abs() < 2e-4, "got {}, exact {}", ys[10_000], exact);
    }

    /// Euler is first order: error ∝ h
    #[test]
    fn convergence_order_euler() {
        let f = |_x: f64, y: f64| y;
        let exact = 1.0_f64.exp();
        let e1 = (solve(&f, 0.0, 1.0, 1.0, 100).1[100] - exact).abs();
        let e2 = (solve(&f, 0.0, 1.0, 1.0, 200).1[200] - exact).abs();
        let ratio = e1 / e2;
        // ratio should be ~2 for first-order method
        assert!(ratio > 1.6 && ratio < 2.6, "convergence ratio = {ratio}, expected ~2");
    }

    /// dy/dx = -y  →  y = e^{-x}
    #[test]
    fn exponential_decay() {
        let f = |_x: f64, y: f64| -y;
        let (_, ys) = solve(&f, 0.0, 1.0, 1.0, 10_000);
        let exact = (-1.0_f64).exp();
        assert!((ys[10_000] - exact).abs() < 1e-4);
    }

    /// dy/dx = 2x  →  y = x² + y0
    #[test]
    fn quadratic_rhs() {
        let f = |x: f64, _y: f64| 2.0 * x;
        let (_, ys) = solve(&f, 0.0, 0.0, 3.0, 10_000);
        let exact = 9.0;
        assert!((ys[10_000] - exact).abs() < 1e-2);
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
        // Euler single step: y = 1 + 1*1 = 2
        assert!((ys[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn negative_direction() {
        // integrate backwards from x=1 to x=0
        let f = |_x, y| y;
        let (_, ys) = solve(&f, 1.0, 1.0_f64.exp(), 0.0, 10_000);
        // exact: y(0) = 1
        assert!((ys[10_000] - 1.0).abs() < 1e-3);
    }

    /// dy/dx = cos(x) → y = sin(x) + C
    #[test]
    fn trigonometric_rhs() {
        let f = |x: f64, _y: f64| x.cos();
        let (_, ys) = solve(&f, 0.0, 0.0, std::f64::consts::PI / 2.0, 10_000);
        let exact = 1.0;
        assert!((ys[10_000] - exact).abs() < 1e-3);
    }
}
