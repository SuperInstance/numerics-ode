//! # numerics-ode
//!
//! Research-grade ordinary differential equation (ODE) solvers implemented in pure Rust
//! with zero external dependencies.
//!
//! ## Solvers
//!
//! - **Euler** — First-order explicit method. Simple but only O(h) accurate.
//! - **RK4** — Classical fourth-order Runge-Kutta. O(h⁴) global error.
//! - **Adams-Bashforth** — Second-order two-step explicit multistep method. O(h²).
//! - **Dormand-Prince** — Embedded RK4(5) with adaptive step-size control. O(h⁴/⁵).
//!
//! ## System support
//!
//! All solvers support both scalar ODEs (`dy/dx = f(x, y)` where `y: f64`)
//! and systems of ODEs (`dy/dx = f(x, &y)` where `y: Vec<f64>`).
//!
//! ## Example
//!
//! ```
//! use numerics_ode::rk4;
//!
//! let f = |_x: f64, y: f64| y; // dy/dx = y
//! let (xs, ys) = rk4::solve(&f, 0.0, 1.0, 1.0, 100);
//! assert!((ys[100] - (1.0_f64).exp()).abs() < 1e-8);
//! ```

pub mod euler;
pub mod rk4;
pub mod adams_bashforth;
pub mod dormand_prince;
pub mod system;

pub use euler::solve as euler_solve;
pub use rk4::solve as rk4_solve;
pub use adams_bashforth::solve as adams_bashforth_solve;
pub use dormand_prince::solve_adaptive as dormand_prince_solve;
