//! Dormand-Prince method (DOPRI5) — embedded RK4(5) with adaptive step-size control.
//!
//! Uses the 7-stage FSAL (First Same As Last) Dormand-Prince coefficients.
//! Step size is adapted based on the local error estimate from the embedded 4th/5th order pair.

/// Result of an adaptive ODE solve.
#[derive(Debug, Clone)]
pub struct AdaptiveResult {
    /// x values at accepted steps.
    pub xs: Vec<f64>,
    /// y values at accepted steps.
    pub ys: Vec<f64>,
    /// Number of rejected steps.
    pub rejected_steps: usize,
    /// Number of accepted steps.
    pub accepted_steps: usize,
}

/// Solve an IVP using the Dormand-Prince RK4(5) method with adaptive step-size control.
///
/// # Arguments
///
/// * `f`       — Right-hand side `dy/dx = f(x, y)`.
/// * `x0`      — Initial x.
/// * `y0`      — Initial y.
/// * `x_end`   — Terminal x.
/// * `h_init`  — Initial step size guess (absolute value used).
/// * `tol`     — Local error tolerance.
/// * `h_min`   — Minimum step size.
/// * `h_max`   — Maximum step size.
/// * `max_steps` — Safety limit on total step attempts.
///
/// # Returns
///
/// `AdaptiveResult` containing solution points and statistics.
#[allow(clippy::too_many_arguments)]
pub fn solve_adaptive(
    f: &dyn Fn(f64, f64) -> f64,
    x0: f64,
    y0: f64,
    x_end: f64,
    h_init: f64,
    tol: f64,
    h_min: f64,
    h_max: f64,
    max_steps: usize,
) -> AdaptiveResult {
    // Dormand-Prince Butcher tableau coefficients
    const A: [[f64; 5]; 5] = [
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0],
        [19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0],
        [9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0],
    ];
    const B5: [f64; 6] = [35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0];
    const E: [f64; 7] = [
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    ];

    let mut xs = vec![x0];
    let mut ys = vec![y0];
    let mut x = x0;
    let mut y = y0;
    // signed step: positive for forward, negative for backward
    let sign = if x_end >= x0 { 1.0 } else { -1.0 };
    let mut h = sign * h_init.abs();
    let h_min_s = sign * h_min.abs();
    let h_max_s = sign * h_max.abs();
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut total = 0usize;

    loop {
        if total >= max_steps {
            break;
        }
        // Don't overshoot
        let remaining = x_end - x;
        if sign * (remaining - h) < 0.0 {
            h = remaining;
        }
        if h.abs() < h_min_s.abs() {
            h = h_min_s;
        }

        let k1 = f(x, y);
        let k2 = f(x + h * A[0][0], y + h * A[0][0] * k1);
        let k3 = f(
            x + h * (A[1][0] + A[1][1]),
            y + h * (A[1][0] * k1 + A[1][1] * k2),
        );
        let k4 = f(
            x + h * (A[2][0] + A[2][1] + A[2][2]),
            y + h * (A[2][0] * k1 + A[2][1] * k2 + A[2][2] * k3),
        );
        let k5 = f(
            x + h * (A[3][0] + A[3][1] + A[3][2] + A[3][3]),
            y + h * (A[3][0] * k1 + A[3][1] * k2 + A[3][2] * k3 + A[3][3] * k4),
        );
        let k6 = f(
            x + h * (A[4][0] + A[4][1] + A[4][2] + A[4][3] + A[4][4]),
            y + h * (A[4][0] * k1 + A[4][1] * k2 + A[4][2] * k3 + A[4][3] * k4 + A[4][4] * k5),
        );

        let y_new = y + h * (B5[0] * k1 + B5[1] * k2 + B5[2] * k3 + B5[3] * k4 + B5[4] * k5 + B5[5] * k6);
        let k7 = f(x + h, y_new);
        let err = h * (E[0] * k1 + E[1] * k2 + E[2] * k3 + E[3] * k4 + E[4] * k5 + E[5] * k6 + E[6] * k7);
        let err_norm = err.abs();
        total += 1;

        if err_norm <= tol || h.abs() <= h_min_s.abs() {
            x += h;
            y = y_new;
            xs.push(x);
            ys.push(y);
            accepted += 1;

            if err_norm > 0.0 {
                let scale = 0.9 * (tol / err_norm).powf(0.2);
                h *= scale.min(5.0);
            } else {
                h *= 2.0;
            }
            if h.abs() > h_max_s.abs() {
                h = h_max_s;
            }

            if (x_end - x).abs() < 1e-14 * x_end.abs().max(1.0) {
                break;
            }
        } else {
            rejected += 1;
            let scale = 0.9 * (tol / err_norm).powf(0.2);
            h *= scale.max(0.1);
            if h.abs() < h_min_s.abs() {
                h = h_min_s;
            }
        }
    }

    AdaptiveResult {
        xs,
        ys,
        rejected_steps: rejected,
        accepted_steps: accepted,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_growth() {
        let f = |_x: f64, y: f64| y;
        let res = solve_adaptive(&f, 0.0, 1.0, 1.0, 0.1, 1e-8, 1e-10, 0.5, 10_000);
        let exact = 1.0_f64.exp();
        let last = *res.ys.last().unwrap();
        assert!((last - exact).abs() < 1e-6, "got {last}, exact {exact}");
    }

    #[test]
    fn constant_rhs() {
        let f = |_x: f64, _y: f64| 0.0;
        let res = solve_adaptive(&f, 0.0, 42.0, 1.0, 0.1, 1e-6, 1e-10, 0.5, 1000);
        let last = *res.ys.last().unwrap();
        assert!((last - 42.0).abs() < 1e-12);
    }

    #[test]
    fn linear_rhs() {
        let f = |_x: f64, _y: f64| 5.0;
        let res = solve_adaptive(&f, 0.0, 0.0, 2.0, 0.1, 1e-8, 1e-10, 0.5, 1000);
        let last = *res.ys.last().unwrap();
        assert!((last - 10.0).abs() < 1e-8);
    }

    #[test]
    fn sinusoidal_rhs() {
        let f = |x: f64, _y: f64| x.cos();
        let res = solve_adaptive(&f, 0.0, 0.0, std::f64::consts::PI / 2.0, 0.1, 1e-10, 1e-12, 0.2, 10_000);
        let last = *res.ys.last().unwrap();
        assert!((last - 1.0).abs() < 1e-8);
    }

    #[test]
    fn exponential_decay() {
        let f = |_x: f64, y: f64| -y;
        let res = solve_adaptive(&f, 0.0, 1.0, 2.0, 0.1, 1e-8, 1e-10, 0.5, 10_000);
        let last = *res.ys.last().unwrap();
        assert!((last - (-2.0_f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn adaptive_reduces_steps_for_easy_problems() {
        let f = |_x: f64, _y: f64| 0.0;
        let res = solve_adaptive(&f, 0.0, 1.0, 10.0, 0.1, 1e-6, 1e-10, 1.0, 10_000);
        assert!(res.accepted_steps < 20, "took {} steps for trivial problem", res.accepted_steps);
    }

    #[test]
    fn quadratic_rhs() {
        let f = |x: f64, _y: f64| 2.0 * x;
        let res = solve_adaptive(&f, 0.0, 0.0, 3.0, 0.1, 1e-10, 1e-12, 0.5, 10_000);
        let last = *res.ys.last().unwrap();
        assert!((last - 9.0).abs() < 1e-8);
    }

    #[test]
    fn result_contains_initial_point() {
        let f = |_x: f64, y: f64| y;
        let res = solve_adaptive(&f, 0.0, 1.0, 1.0, 0.1, 1e-6, 1e-10, 0.5, 1000);
        assert_eq!(res.xs[0], 0.0);
        assert_eq!(res.ys[0], 1.0);
    }

    #[test]
    fn tol_affects_accuracy() {
        let f = |_x: f64, y: f64| y;
        let exact = 1.0_f64.exp();
        let res_loose = solve_adaptive(&f, 0.0, 1.0, 1.0, 0.1, 1e-4, 1e-10, 0.5, 10_000);
        let res_tight = solve_adaptive(&f, 0.0, 1.0, 1.0, 0.1, 1e-12, 1e-14, 0.5, 10_000);
        let e_loose = (res_loose.ys.last().unwrap() - exact).abs();
        let e_tight = (res_tight.ys.last().unwrap() - exact).abs();
        assert!(e_tight < e_loose, "tight tol should be more accurate: {e_tight} vs {e_loose}");
    }

    #[test]
    fn stiff_problem() {
        let f = |_x: f64, y: f64| -50.0 * y;
        let res = solve_adaptive(&f, 0.0, 1.0, 0.1, 0.001, 1e-6, 1e-8, 0.01, 100_000);
        let last = *res.ys.last().unwrap();
        let exact = (-5.0_f64).exp();
        assert!((last - exact).abs() < 1e-4, "got {last}, exact {exact}");
    }

    #[test]
    fn negative_direction() {
        let f = |_x: f64, y: f64| y;
        let res = solve_adaptive(&f, 1.0, 1.0_f64.exp(), 0.0, 0.1, 1e-8, 1e-12, 0.5, 10_000);
        let last = *res.ys.last().unwrap();
        assert!((last - 1.0).abs() < 1e-5, "got {last}");
    }
}
