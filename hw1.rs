fn main() {
    println!("Hello, world!");

    // f(x) = x^3
    // f'(x) = 3x^2
    // f'(6) = 3*6^2 = 108
    let f = &|x: &[f64]| x.iter().map(|&i| i.powi(3)).collect();
    let derivative = numerical_derivative(f, &[6.0], None);
    println!("Expected: 108, calculated: {derivative:?}");

    // f(x) = x^3 + 4x^2 - 12
    // f'(x) = 3x^2 + 8x
    // f'(2) = 3*2^2 + 8*2 = 28
    let f = &|x: &[f64]| {
        x.iter()
            .map(|&i| i.powi(3) + 4.0 * i.powi(2) - 12.0)
            .collect()
    };
    let derivative = numerical_derivative(f, &[2.0], Some(1e-5));
    println!("Expected: 28, calculated: {derivative:?}");
}

/// (f(x + h) - f(x)) / h
fn numerical_derivative(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], h: Option<f64>) -> Vec<f64> {
    let h = h.unwrap_or(1e-5);
    let x_h: Vec<f64> = x.iter().map(|&i| i + h).collect(); // x + h
    f(&x_h)
        .iter()
        .zip(f(x).iter())
        .map(|(a, b)| (a - b) / h)
        .collect()
}
