//! Adams-Bashforth 2-step explicit multistep method.
//!
//! Requires a bootstrap step (computed with RK4) to start.
//!
//! ```text
//! y_{n+1} = y_n + (h/2)(3·f(x_n, y_n) - f(x_{n-1}, y_{n-1}))
//! ```
//!
//! Local truncation error: O(h³), global: O(h²).

/// Solve an IVP using the 2-step Adams-Bashforth method.
///
/// The first step is bootstrapped with a single RK4 step.
///
/// # Arguments
///
/// * `f`     — Right-hand side.
/// * `x0`    — Initial x.
/// * `y0`    — Initial y.
/// * `x_end` — Terminal x.
/// * `n`     — Number of steps (≥ 2 for AB2 to be used; if `n == 1`, falls back to Euler).
///
/// # Returns
///
/// `(xs, ys)` — all computed points.
pub fn solve(f: &dyn Fn(f64, f64) -> f64, x0: f64, y0: f64, x_end: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 1, "number of steps must be ≥ 1");
    let h = (x_end - x0) / n as f64;
    let mut xs = Vec::with_capacity(n + 1);
    let mut ys = Vec::with_capacity(n + 1);
    xs.push(x0);
    ys.push(y0);

    if n == 1 {
        // fallback: single Euler step
        let y1 = y0 + h * f(x0, y0);
        xs.push(x0 + h);
        ys.push(y1);
        return (xs, ys);
    }

    // Bootstrap: use RK4 for the first step
    let x0_val = x0;
    let y0_val = y0;
    let k1 = f(x0_val, y0_val);
    let k2 = f(x0_val + h / 2.0, y0_val + h * k1 / 2.0);
    let k3 = f(x0_val + h / 2.0, y0_val + h * k2 / 2.0);
    let k4 = f(x0_val + h, y0_val + h * k3);
    let y1 = y0_val + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    let x1 = x0_val + h;
    xs.push(x1);
    ys.push(y1);

    let mut f_prev = f(x0_val, y0_val);
    let mut x_curr = x1;
    let mut y_curr = y1;

    for _ in 1..n {
        let f_curr = f(x_curr, y_curr);
        let y_next = y_curr + (h / 2.0) * (3.0 * f_curr - f_prev);
        x_curr += h;
        xs.push(x_curr);
        ys.push(y_next);
        f_prev = f_curr;
        y_curr = y_next;
    }

    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_growth() {
        let f = |_x: f64, y: f64| y;
        let (_, ys) = solve(&f, 0.0, 1.0, 1.0, 100);
        let exact = 1.0_f64.exp();
        assert!((ys[100] - exact).abs() < 5e-4);
    }

    #[test]
    fn convergence_order() {
        let f = |_x: f64, y: f64| y;
        let exact = 1.0_f64.exp();
        let e1 = (solve(&f, 0.0, 1.0, 1.0, 50).1[50] - exact).abs();
        let e2 = (solve(&f, 0.0, 1.0, 1.0, 100).1[100] - exact).abs();
        let ratio = e1 / e2;
        // AB2 is second order: ratio should be ~4 (2²)
        assert!(ratio > 2.5 && ratio < 7.0, "ratio = {ratio}");
    }

    #[test]
    fn constant_rhs() {
        let f = |_x, _y| 0.0;
        let (_, ys) = solve(&f, 0.0, 10.0, 1.0, 10);
        for &y in &ys {
            assert!((y - 10.0).abs() < 1e-12);
        }
    }

    #[test]
    fn linear_rhs() {
        let f = |_x, _y| 3.0;
        let (_, ys) = solve(&f, 0.0, 0.0, 2.0, 100);
        assert!((ys[100] - 6.0).abs() < 1e-8);
    }

    #[test]
    fn sinusoidal_rhs() {
        let f = |x: f64, _y: f64| x.cos();
        let (_, ys) = solve(&f, 0.0, 0.0, std::f64::consts::PI / 2.0, 200);
        assert!((ys[200] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn exponential_decay() {
        let f = |_x: f64, y: f64| -y;
        let (_, ys) = solve(&f, 0.0, 1.0, 1.0, 200);
        assert!((ys[200] - (-1.0_f64).exp()).abs() < 1e-5);
    }

    #[test]
    fn single_step_fallback() {
        let f = |_x, y| y;
        let (xs, ys) = solve(&f, 0.0, 1.0, 1.0, 1);
        assert_eq!(xs.len(), 2);
        assert_eq!(ys.len(), 2);
    }

    #[test]
    #[should_panic(expected = "number of steps must be ≥ 1")]
    fn panics_on_zero_steps() {
        let f = |_x, _y| 0.0;
        solve(&f, 0.0, 1.0, 1.0, 0);
    }

    #[test]
    fn quadratic_rhs() {
        let f = |x: f64, _y: f64| 2.0 * x;
        let (_, ys) = solve(&f, 0.0, 0.0, 3.0, 200);
        assert!((ys[200] - 9.0).abs() < 1e-4);
    }

    #[test]
    fn negative_direction() {
        let f = |_x, y| y;
        let (_, ys) = solve(&f, 1.0, 1.0_f64.exp(), 0.0, 200);
        assert!((ys[200] - 1.0).abs() < 1e-3);
    }
}
