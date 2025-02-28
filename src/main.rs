fn main() {
    println!("Hello, world!");

    // f(x) = x^3
    // f'(x) = 3x^2
    // f'(6) = 3*6^2 = 108
    print!("f(x) = x^3");
    let f = |x: &[f64]| x.iter().map(|&i| i.powi(3)).collect();
    let derivative = numerical_derivative(&f, &[6.0], None);
    test(derivative, &[&[108.0]]);

    // f(x) = x^3 + 4x^2 - 12
    // f'(x) = 3x^2 + 8x
    // f'(2) = 3*2^2 + 8*2 = 28
    print!("f(x) = x^3 + 4x^2 - 12");
    let f = |x: &[f64]| {
        x.iter()
            .map(|&i| i.powi(3) + 4.0 * i.powi(2) - 12.0)
            .collect()
    };
    let derivative = numerical_derivative(&f, &[2.0], Some(1e-5));
    test(derivative, &[&[28.0]]);

    // f([x, y]) = [sin(x) + cos(y), cos(x) - sin(y)]
    // f'([x, y]) = [[cos(x), -sin(x)], [-sin(y), -cos(y)]]
    // f'([π/4, π/3]) = [[cos(π/4), -sin(π/4)], [-sin(π/3), -cos(π/3)]] = [[√2/2, -√2/2], [-√3/2, -1/2]]
    print!("f([x, y]) = [sin(x) + cos(y), cos(x) - sin(y)]");
    let f = |x: &[f64]| vec![x[0].sin() + x[1].cos(), x[0].cos() - x[1].sin()];
    let v = [std::f64::consts::FRAC_PI_4, std::f64::consts::FRAC_PI_3];
    let derivative = numerical_derivative(&f, &v, None);
    let expected: &[&[f64]] = &[&[0.7071, -0.7071], &[-0.8660, -0.5]];
    test(derivative, expected);

    println!("All tests passed 🎉");
}

/// Pretty print test results
fn test(actual: Vec<Vec<f64>>, expected: &[&[f64]]) {
    for (a, e) in actual.iter().zip(expected.iter()) {
        for (a, e) in a.iter().zip(e.iter()) {
            // assert!((a - e).abs() < 1e-3);
            if (a - e).abs() > 1e-3 {
                println!(" FAIL ❌");
                println!(" expected: {}, got: {}", e, a);
                std::process::exit(1);
            }
        }
    }
    println!(" OK 👍");
}

fn numerical_derivative(
    f: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    h: Option<f64>,
) -> Vec<Vec<f64>> {
    let h = h.unwrap_or(1e-7);
    let f_x = f(x);
    let mut jacobian = vec![vec![0.0; x.len()]; x.len()]; // TODO: no need to create here, since it will be overwritten
    for i in 0..x.len() {
        let mut direction = vec![0.0; x.len()];
        direction[i] = h;
        let x_h: Vec<f64> = x
            .iter()
            .zip(direction.iter())
            .map(|(&i, h)| i + h)
            .collect();
        jacobian[i] = f(&x_h)
            .iter()
            .zip(f_x.iter())
            .map(|(a, b)| (a - b) / h)
            .collect();
    }
    jacobian
}
