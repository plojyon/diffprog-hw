fn main() {
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

    // f([x, y, z]) = [x^2 + y^2 + z^2, x + y + z]
    // f'([x, y, z]) = [[2x, 1], [2y, 1], [2z, 1]]
    // f'([1, 2, 3]) = [[2, 1], [4, 1], [6, 1]]
    print!("f([x, y, z]) = [x^2 + y^2 + z^2, x + y + z]");
    let f = |x: &[f64]| {
        vec![
            x[0].powi(2) + x[1].powi(2) + x[2].powi(2),
            x[0] + x[1] + x[2],
        ]
    };
    let v = [1.0, 2.0, 3.0];
    let derivative = numerical_derivative(&f, &v, None);
    let expected: &[&[f64]] = &[&[2.0, 1.0], &[4.0, 1.0], &[6.0, 1.0]];
    test(derivative, expected);

    // f([x, y, z, w]) = [sin(x^2)w + y^2z, xwz^3tan(y), x, 69, -5z]
    // f'([x, y, z, w])^T = [[2xcos(x^2)w, wz^3tan(y), 1, 0, 0],[2yz, xwz^3sec^2(y), 0, 0, 0],[sin(x^2), 3xwz^2tan(y), 0, 0, -5],[0, xz^3tan(y), 0, 0, 0]]
    // f'([π/4, π/3, 2, 3])^T = [[3.84391697914949, 41.5692193816531, 1.0, 0, 0],[4.18879020478639, 75.398223686155, 0, 0, 0],[1.09662271123215, 48.9725828343239, 0, 0, -5.0],[0.578468789354558, 10.8827961854053, 0, 0, 0]]
    print!("f([x, y, z, w]) = [sin(x^2)w + y^2z, xwz^3tan(y), x, 69, -5z]");
    let f = |x: &[f64]| {
        vec![
            x[3] * (x[0].powi(2).sin()) + x[1].powi(2) * x[2],
            x[0] * x[3] * x[2].powi(3) * x[1].tan(),
            x[0],
            69.0,
            -5.0 * x[2],
        ]
    };
    let v = [
        std::f64::consts::FRAC_PI_4,
        std::f64::consts::FRAC_PI_3,
        2.0,
        3.0,
    ];
    let derivative = numerical_derivative(&f, &v, None);
    let expected: &[&[f64]] = &[
        &[3.84391697914949, 41.5692193816531, 1.0, 0.0, 0.0],
        &[4.18879020478639, 75.398223686155, 0.0, 0.0, 0.0],
        &[1.09662271123215, 48.9725828343239, 0.0, 0.0, -5.0],
        &[0.578468789354558, 10.8827961854053, 0.0, 0.0, 0.0],
    ];
    test(derivative, expected);

    println!("All tests passed 🎉");
}

/// Pretty print test results
fn test(actual: Vec<Vec<f64>>, expected: &[&[f64]]) {
    let mut err = 0.0;
    for (a, e) in actual.iter().zip(expected.iter()) {
        for (a, e) in a.iter().zip(e.iter()) {
            // assert!((a - e).abs() < 1e-3);
            if (a - e).abs() > 1e-3 {
                println!(" FAIL ❌");
                println!(" expected: {}, got: {}", e, a);
                std::process::exit(1);
            }
            err += (a - e).abs();
        }
    }
    println!(" OK 👍 ({})", err);
}

fn numerical_derivative(
    f: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &[f64],
    h: Option<f64>,
) -> Vec<Vec<f64>> {
    let f_x = f(x);
    let mut jacobian = vec![vec![0.0; 0]; x.len()];
    for i in 0..x.len() {
        let h = h.unwrap_or(f64::sqrt(f64::EPSILON) * x[i]);
        let mut x_h = x.to_vec();
        x_h[i] += h;
        jacobian[i] = f(&x_h)
            .iter()
            .zip(f_x.iter())
            .map(|(a, b)| (a - b) / h)
            .collect();
    }
    jacobian
}
