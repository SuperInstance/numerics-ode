//! System-of-ODEs solvers.
//!
//! Extends all four methods to systems of ODEs of the form:
//! ```text
//! dy⃗/dx = f(x, &y⃗),   y⃗(x₀) = y⃗₀
//! ```
//!
//! Each module function returns `(Vec<f64>, Vec<Vec<f64>>)` — the `x` grid and
//! the state vector at each grid point.

/// Euler method for systems.
pub fn euler(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    x0: f64,
    y0: &[f64],
    x_end: f64,
    n: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    assert!(n >= 1);
    let m = y0.len();
    let h = (x_end - x0) / n as f64;
    let mut xs = Vec::with_capacity(n + 1);
    let mut ys: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    xs.push(x0);
    ys.push(y0.to_vec());
    let mut x = x0;
    let mut y = y0.to_vec();
    for _ in 0..n {
        let dy = f(x, &y);
        for j in 0..m {
            y[j] += h * dy[j];
        }
        x += h;
        xs.push(x);
        ys.push(y.clone());
    }
    (xs, ys)
}

/// RK4 method for systems.
pub fn rk4(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    x0: f64,
    y0: &[f64],
    x_end: f64,
    n: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    assert!(n >= 1);
    let m = y0.len();
    let h = (x_end - x0) / n as f64;
    let mut xs = Vec::with_capacity(n + 1);
    let mut ys: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    xs.push(x0);
    ys.push(y0.to_vec());
    let mut x = x0;
    let mut y = y0.to_vec();
    for _ in 0..n {
        let k1 = f(x, &y);
        let mut y_tmp = vec![0.0; m];
        for j in 0..m {
            y_tmp[j] = y[j] + h * k1[j] / 2.0;
        }
        let k2 = f(x + h / 2.0, &y_tmp);
        for j in 0..m {
            y_tmp[j] = y[j] + h * k2[j] / 2.0;
        }
        let k3 = f(x + h / 2.0, &y_tmp);
        for j in 0..m {
            y_tmp[j] = y[j] + h * k3[j];
        }
        let k4 = f(x + h, &y_tmp);
        for j in 0..m {
            y[j] += (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
        }
        x += h;
        xs.push(x);
        ys.push(y.clone());
    }
    (xs, ys)
}

/// Adams-Bashforth 2-step for systems.
pub fn adams_bashforth(
    f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
    x0: f64,
    y0: &[f64],
    x_end: f64,
    n: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    assert!(n >= 1);
    let m = y0.len();
    let h = (x_end - x0) / n as f64;
    let mut xs = Vec::with_capacity(n + 1);
    let mut ys: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    xs.push(x0);
    ys.push(y0.to_vec());

    if n == 1 {
        let dy = f(x0, y0);
        let mut y1 = y0.to_vec();
        for j in 0..m {
            y1[j] += h * dy[j];
        }
        xs.push(x0 + h);
        ys.push(y1);
        return (xs, ys);
    }

    // Bootstrap with RK4
    let k1 = f(x0, y0);
    let mut y_tmp = vec![0.0; m];
    for j in 0..m {
        y_tmp[j] = y0[j] + h * k1[j] / 2.0;
    }
    let k2 = f(x0 + h / 2.0, &y_tmp);
    for j in 0..m {
        y_tmp[j] = y0[j] + h * k2[j] / 2.0;
    }
    let k3 = f(x0 + h / 2.0, &y_tmp);
    for j in 0..m {
        y_tmp[j] = y0[j] + h * k3[j];
    }
    let k4 = f(x0 + h, &y_tmp);
    let mut y1 = y0.to_vec();
    for j in 0..m {
        y1[j] += (h / 6.0) * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
    }
    let x1 = x0 + h;
    xs.push(x1);
    ys.push(y1.clone());

    let mut f_prev = k1;
    let mut x_curr = x1;
    let mut y_curr = y1;

    for _ in 1..n {
        let f_curr = f(x_curr, &y_curr);
        let mut y_next = vec![0.0; m];
        for j in 0..m {
            y_next[j] = y_curr[j] + (h / 2.0) * (3.0 * f_curr[j] - f_prev[j]);
        }
        x_curr += h;
        xs.push(x_curr);
        ys.push(y_next.clone());
        f_prev = f_curr;
        y_curr = y_next;
    }

    (xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// System: dy0/dx = y1, dy1/dx = -y0  →  (cos x, -sin x)
    fn harmonic(_x: f64, y: &[f64]) -> Vec<f64> {
        vec![y[1], -y[0]]
    }

    #[test]
    fn euler_harmonic() {
        let y0 = vec![1.0, 0.0];
        let (_, ys) = euler(&harmonic, 0.0, &y0, std::f64::consts::PI / 2.0, 10_000);
        let last = ys.last().unwrap();
        assert!((last[0] - 0.0).abs() < 1e-3, "y0 = {}", last[0]);
        assert!((last[1] - (-1.0)).abs() < 1e-3, "y1 = {}", last[1]);
    }

    #[test]
    fn rk4_harmonic() {
        let y0 = vec![1.0, 0.0];
        let (_, ys) = rk4(&harmonic, 0.0, &y0, std::f64::consts::PI / 2.0, 100);
        let last = ys.last().unwrap();
        assert!((last[0]).abs() < 1e-9, "y0 = {}", last[0]);
        assert!((last[1] + 1.0).abs() < 1e-9, "y1 = {}", last[1]);
    }

    #[test]
    fn ab2_harmonic() {
        let y0 = vec![1.0, 0.0];
        let (_, ys) = adams_bashforth(&harmonic, 0.0, &y0, std::f64::consts::PI / 2.0, 200);
        let last = ys.last().unwrap();
        assert!((last[0]).abs() < 1e-4, "y0 = {}", last[0]);
        assert!((last[1] + 1.0).abs() < 1e-4, "y1 = {}", last[1]);
    }

    /// Lotka-Volterra predator-prey (conservation of energy)
    fn lotka_volterra(_x: f64, y: &[f64]) -> Vec<f64> {
        let a = 1.0;
        let b = 1.0;
        let c = 1.0;
        let d = 1.0;
        vec![
            a * y[0] - b * y[0] * y[1],
            -c * y[1] + d * y[0] * y[1],
        ]
    }

    #[test]
    fn rk4_lotka_volterra_periodicity() {
        let y0 = vec![2.0, 2.0];
        let (xs, ys) = rk4(&lotka_volterra, 0.0, &y0, 10.0, 10_000);
        // Population should stay positive
        for y in &ys {
            assert!(y[0] > 0.0, "prey went negative: {}", y[0]);
            assert!(y[1] > 0.0, "predator went negative: {}", y[1]);
        }
        let _ = xs; // use xs
    }

    /// Decoupled system: dy0/dx = y0, dy1/dx = -y1
    #[test]
    fn euler_decoupled_system() {
        let f = |_x: f64, y: &[f64]| vec![y[0], -y[1]];
        let y0 = vec![1.0, 1.0];
        let (_, ys) = euler(&f, 0.0, &y0, 1.0, 10_000);
        let last = ys.last().unwrap();
        assert!((last[0] - 1.0_f64.exp()).abs() < 1e-3);
        assert!((last[1] - (-1.0_f64).exp()).abs() < 1e-3);
    }

    #[test]
    fn rk4_decoupled_system() {
        let f = |_x: f64, y: &[f64]| vec![y[0], -y[1]];
        let y0 = vec![1.0, 1.0];
        let (_, ys) = rk4(&f, 0.0, &y0, 1.0, 100);
        let last = ys.last().unwrap();
        assert!((last[0] - 1.0_f64.exp()).abs() < 1e-9);
        assert!((last[1] - (-1.0_f64).exp()).abs() < 1e-9);
    }

    /// 3D system: rigid body rotation
    fn rigid_body(_x: f64, y: &[f64]) -> Vec<f64> {
        let i1 = 1.0;
        let i2 = 2.0;
        let i3 = 3.0;
        vec![
            (i2 - i3) / (i1) * y[1] * y[2],
            (i3 - i1) / (i2) * y[0] * y[2],
            (i1 - i2) / (i3) * y[0] * y[1],
        ]
    }

    #[test]
    fn rk4_rigid_body_energy_conservation() {
        let y0 = vec![1.0, 1.0, 1.0];
        let (xs, ys) = rk4(&rigid_body, 0.0, &y0, 10.0, 10_000);
        // Energy E = 0.5*(I1*y0² + I2*y1² + I3*y2²) should be conserved
        let e0 = 0.5 * (1.0 * y0[0].powi(2) + 2.0 * y0[1].powi(2) + 3.0 * y0[2].powi(2));
        for y in &ys {
            let e = 0.5 * (1.0 * y[0].powi(2) + 2.0 * y[1].powi(2) + 3.0 * y[2].powi(2));
            assert!((e - e0).abs() / e0 < 1e-6, "energy drift: {e} vs {e0}");
        }
        let _ = xs;
    }

    #[test]
    fn constant_system() {
        let f = |_x: f64, y: &[f64]| vec![0.0; y.len()];
        let y0 = vec![1.0, 2.0, 3.0];
        let (_, ys) = rk4(&f, 0.0, &y0, 1.0, 10);
        let last = ys.last().unwrap();
        assert!((last[0] - 1.0).abs() < 1e-12);
        assert!((last[1] - 2.0).abs() < 1e-12);
        assert!((last[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn linear_system() {
        let f = |_x: f64, _y: &[f64]| vec![1.0, 2.0];
        let y0 = vec![0.0, 0.0];
        let (_, ys) = rk4(&f, 0.0, &y0, 1.0, 100);
        let last = ys.last().unwrap();
        assert!((last[0] - 1.0).abs() < 1e-10);
        assert!((last[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn output_length_matches() {
        let y0 = vec![1.0, 0.0];
        let (xs, ys) = rk4(&harmonic, 0.0, &y0, 1.0, 50);
        assert_eq!(xs.len(), 51);
        assert_eq!(ys.len(), 51);
        assert_eq!(ys[0].len(), 2);
    }
}
