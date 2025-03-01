use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Mul};

// Debugging expression simplification logic
const DISABLE_SIMPLIFICATION: bool = false;
const DEBUG_SIMPLIFICATION: bool = false;

#[derive(Debug, Clone)]
enum Operation {
    Add,
    Mul,
    Pow,
    Sin,
    Cos,
    Log,
    Var(String),
    Const(f64),
}

#[derive(Clone)]
struct Node {
    op: Operation,
    args: Vec<Box<Node>>,
}
impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.op {
            Operation::Var(name) => write!(f, "{}", name),
            Operation::Const(value) => write!(f, "{}", value),
            _ => {
                let arguments = self
                    .args
                    .iter()
                    .map(|arg| format!("{:?}", arg))
                    .collect::<Vec<String>>()
                    .join(", ");
                write!(f, "{:?}({})", self.op, arguments)
            }
        }
    }
}

impl Node {
    fn new(op: Operation, args: Vec<Box<Node>>) -> Self {
        if DISABLE_SIMPLIFICATION {
            return Self { op, args };
        } else {
            let mut node = Self { op, args };
            node.simplify()
        }
    }

    #[allow(dead_code)]
    fn simplify_tree(&mut self) -> Node {
        let orig = format!("{:?}", self);
        for arg in &mut self.args {
            *arg = Box::new(arg.simplify_tree());
        }
        let b = self.simplify();
        if DEBUG_SIMPLIFICATION {
            println!("Simplified {} to {:?}", orig, b);
        }
        b
    }

    fn simplify(&mut self) -> Node {
        let op = &self.op;
        let mut args = self.args.clone();

        fn eq(a: &Operation, b: f64) -> bool {
            matches!(a, Operation::Const(value) if (value - b).abs() < 1e-5)
        }

        match op {
            // a + 0 = a
            // evaluate const + const
            Operation::Add => {
                if eq(&args[0].op, 0.0) {
                    return *args.remove(1);
                }
                if eq(&args[1].op, 0.0) {
                    return *args.remove(0);
                }
                if let Operation::Const(a) = args[0].op {
                    if let Operation::Const(b) = args[1].op {
                        return c(a + b);
                    }
                }
            }
            // a * 1 = a
            // a * 0 = 0
            // evaluate const * const
            Operation::Mul => {
                if eq(&args[0].op, 1.0) {
                    return *args.remove(1);
                }
                if eq(&args[1].op, 1.0) {
                    return *args.remove(0);
                }
                if eq(&args[0].op, 0.0) || eq(&args[1].op, 0.0) {
                    return c(0.0);
                }
                if let Operation::Const(a) = args[0].op {
                    if let Operation::Const(b) = args[1].op {
                        return c(a * b);
                    }
                }
            }
            // a ^ 1 = a
            // a ^ 0 = 1
            // evaluate const ^ const
            Operation::Pow => {
                if eq(&args[1].op, 1.0) {
                    return *args.remove(0);
                }
                if eq(&args[1].op, 0.0) {
                    return c(1.0);
                }
                if let Operation::Const(a) = args[0].op {
                    if let Operation::Const(b) = args[1].op {
                        return c(a.powf(b));
                    }
                }
            }
            // evaluate sin(const)
            Operation::Sin => {
                if let Operation::Const(value) = args[0].op {
                    return c(value.sin());
                }
            }
            // evaluate cos(const)
            Operation::Cos => {
                if let Operation::Const(value) = args[0].op {
                    return c(value.cos());
                }
            }
            // evaluate log_const(const)
            Operation::Log => {
                if let Operation::Const(base) = args[0].op {
                    if let Operation::Const(value) = args[1].op {
                        return c(value.log(base));
                    }
                }
            }
            Operation::Var(_) => (),
            Operation::Const(_) => (),
        };
        Self {
            op: op.clone(),
            args,
        }
    }

    /// Compute partial derivative wrt. variable
    fn partial_derivative(&mut self, variable: &String) -> Node {
        match &self.op {
            Operation::Var(name) => {
                if name == variable {
                    c(1.0)
                } else {
                    c(0.0)
                }
            }
            Operation::Const(_) => c(0.0),
            Operation::Add => {
                // (a + b)' = a' + b'
                let da = self.args[0].partial_derivative(variable);
                let db = self.args[1].partial_derivative(variable);
                da + db
            }
            Operation::Mul => {
                // (a * b)' = a' * b + a * b'
                let da = self.args[0].partial_derivative(variable);
                let db = self.args[1].partial_derivative(variable);
                let a = *self.args[0].clone();
                let b = *self.args[1].clone();
                da * b + a * db
            }
            Operation::Pow => {
                // (a ^ b)' = a ^ b * (b' * ln(a) + b * a' * a^-1)
                let da = self.args[0].partial_derivative(variable);
                let db = self.args[1].partial_derivative(variable);
                let a = *self.args[0].clone();
                let b = *self.args[1].clone();
                pow(a.clone(), b.clone()) * (db * ln(a.clone()) + b * da * pow(a.clone(), c(-1.0)))
            }
            Operation::Sin => {
                // (sin(a))' = cos(a) * a'
                let da = self.args[0].partial_derivative(variable);
                let a = *self.args[0].clone();
                cos(a) * da
            }
            Operation::Cos => {
                // (cos(a))' = -sin(a) * a'
                let da = self.args[0].partial_derivative(variable);
                let a = *self.args[0].clone();
                -1.0 * sin(a) * da
            }
            Operation::Log => {
                // (log_a(b))' = (b'/b * ln(a) - a'/a * ln(b)) / ln(a)^2
                let da = self.args[0].partial_derivative(variable);
                let db = self.args[1].partial_derivative(variable);
                let a = *self.args[0].clone();
                let b = *self.args[1].clone();
                (db * pow(b.clone(), c(-1.0)) * ln(a.clone())
                    + -1.0 * da * pow(a.clone(), c(-1.0)) * ln(b.clone()))
                    * pow(ln(a.clone()), c(-2.0))
            }
        }
    }

    fn evaluate(&self, variables: &HashMap<String, f64>) -> f64 {
        let args = self
            .args
            .iter()
            .map(|arg| arg.evaluate(variables))
            .collect::<Vec<f64>>();
        match &self.op {
            Operation::Var(name) => {
                if let Some(value) = variables.get(name) {
                    *value
                } else {
                    panic!("Variable {} not found", name);
                }
            }
            Operation::Const(value) => *value,
            Operation::Add => args[0] + args[1],
            Operation::Mul => args[0] * args[1],
            Operation::Pow => args[0].powf(args[1]),
            Operation::Sin => args[0].sin(),
            Operation::Cos => args[0].cos(),
            Operation::Log => args[1].log(args[0]),
        }
    }
}

#[allow(unreachable_code)]
fn main() {
    // f(x, y) = 3x + 4y + 5
    let x = var("x");
    let y = var("y");
    let mut f = 3.0 * x + 4.0 * y + 5.0;
    let df_dx = f.partial_derivative(&"x".to_string());
    let df_dy = f.partial_derivative(&"y".to_string());
    println!("f = 3x + 4y + 5 = {:?}", f);
    println!("df/dx = {:?}", df_dx);
    println!("df/dy = {:?}", df_dy);
    let mut x_0 = HashMap::new();
    x_0.insert("x".to_string(), 1.0);
    x_0.insert("y".to_string(), 2.0);
    println!("f(1, 2) = {}", f.evaluate(&x_0));
    println!("df/dx(1, 2) = {}", df_dx.evaluate(&x_0));
    println!("df/dy(1, 2) = {}", df_dy.evaluate(&x_0));
    println!();

    // f(x, y) = 3xy + 5
    let x = var("x");
    let y = var("y");
    let mut f = 3.0 * x * y + 5.0;
    let df_dx = f.partial_derivative(&"x".to_string());
    let df_dy = f.partial_derivative(&"y".to_string());
    println!("f = 3xy + 5 = {:?}", f);
    println!("df/dx = {:?}", df_dx);
    println!("df/dy = {:?}", df_dy);
    let mut x_0 = HashMap::new();
    x_0.insert("x".to_string(), 1.0);
    x_0.insert("y".to_string(), 2.0);
    println!("f(1, 2) = {}", f.evaluate(&x_0));
    println!("df/dx(1, 2) = {}", df_dx.evaluate(&x_0));
    println!("df/dy(1, 2) = {}", df_dy.evaluate(&x_0));
    println!();

    // f(x, y, z) = 5x + 3y + 4xyz
    let x = var("x");
    let y = var("y");
    let z = var("z");
    let mut f = 5.0 * x.clone() + 3.0 * y.clone() + 4.0 * x.clone() * y.clone() * z;
    let df_dx = f.partial_derivative(&"x".to_string());
    let df_dy = f.partial_derivative(&"y".to_string());
    let df_dz = f.partial_derivative(&"z".to_string());
    println!("f = 5x + 3y + 4xyz = {:?}", f);
    println!("df/dx = {:?}", df_dx);
    println!("df/dy = {:?}", df_dy);
    println!("df/dz = {:?}", df_dz);
    let mut x_0 = HashMap::new();
    x_0.insert("x".to_string(), 1.0);
    x_0.insert("y".to_string(), 2.0);
    x_0.insert("z".to_string(), 3.0);
    println!("f(1, 2, 3) = {}", f.evaluate(&x_0));
    println!("df/dx(1, 2, 3) = {}", df_dx.evaluate(&x_0));
    println!("df/dy(1, 2, 3) = {}", df_dy.evaluate(&x_0));
    println!("df/dz(1, 2, 3) = {}", df_dz.evaluate(&x_0));
    println!();

    // f(x, y) = 2sin(x) + 3cos(y)
    let x = var("x");
    let y = var("y");
    let mut f = 2.0 * sin(x) + 3.0 * cos(y);
    let df_dx = f.partial_derivative(&"x".to_string());
    let df_dy = f.partial_derivative(&"y".to_string());
    println!("f = 2sin(x) + 3cos(y) = {:?}", f);
    println!("df/dx = {:?}", df_dx);
    println!("df/dy = {:?}", df_dy);
    let mut x_0 = HashMap::new();
    x_0.insert("x".to_string(), 1.0);
    x_0.insert("y".to_string(), 2.0);
    println!("f(1, 2) = {}", f.evaluate(&x_0));
    println!("df/dx(1, 2) = {}", df_dx.evaluate(&x_0));
    println!("df/dy(1, 2) = {}", df_dy.evaluate(&x_0));
    println!();

    // f(x, y) = 2log_3(x) + ln(y)
    let x = var("x");
    let y = var("y");
    let mut f = 2.0 * log(c(3.0), x) + ln(y);
    let df_dx = f.partial_derivative(&"x".to_string());
    let df_dy = f.partial_derivative(&"y".to_string());
    println!("f = 2log_3(x) + ln(y) = {:?}", f);
    println!("df/dx = {:?}", df_dx);
    println!("df/dy = {:?}", df_dy);
    let mut x_0 = HashMap::new();
    x_0.insert("x".to_string(), 1.0);
    x_0.insert("y".to_string(), 2.0);
    println!("f(1, 2) = {}", f.evaluate(&x_0));
    println!("df/dx(1, 2) = {}", df_dx.evaluate(&x_0));
    println!("df/dy(1, 2) = {}", df_dy.evaluate(&x_0));
    println!();

    // f(x, y) = tan(ln(x/y))
    let x = var("x");
    let y = var("y");
    let mut f =
        sin(ln(x.clone() * pow(y.clone(), c(-1.0)))) * pow(cos(ln(x * pow(y, c(-1.0)))), c(-1.0));
    let df_dx = f.partial_derivative(&"x".to_string());
    let df_dy = f.partial_derivative(&"y".to_string());
    println!("f = tan(ln(x/y)) = {:?}", f);
    println!("df/dx = {:?}", df_dx);
    println!("df/dy = {:?}", df_dy);
    let mut x_0 = HashMap::new();
    x_0.insert("x".to_string(), 1.0);
    x_0.insert("y".to_string(), 2.0);
    println!("f(1, 2) = {}", f.evaluate(&x_0));
    println!("df/dx(1, 2) = {}", df_dx.evaluate(&x_0));
    println!("df/dy(1, 2) = {}", df_dy.evaluate(&x_0));
    println!();
}

////////////////////
/// Constructors ///
////////////////////
fn c(value: f64) -> Node {
    Node::new(Operation::Const(value), vec![])
}
fn pow(a: Node, b: Node) -> Node {
    Node::new(Operation::Pow, vec![Box::new(a), Box::new(b)])
}
fn log(base: Node, value: Node) -> Node {
    Node::new(Operation::Log, vec![Box::new(base), Box::new(value)])
}
fn ln(value: Node) -> Node {
    log(c(std::f64::consts::E), value)
}
fn sin(value: Node) -> Node {
    Node::new(Operation::Sin, vec![Box::new(value)])
}
fn cos(value: Node) -> Node {
    Node::new(Operation::Cos, vec![Box::new(value)])
}
fn var(name: &str) -> Node {
    Node::new(Operation::Var(name.to_string()), vec![])
}

////////////////////////////
/// Operator overloading ///
////////////////////////////

macro_rules! impl_op {
    ($trait:ident, $method:ident, $op:expr) => {
        impl $trait for Node {
            type Output = Node;

            fn $method(self, other: Self) -> Self {
                Node::new($op, vec![Box::new(self), Box::new(other)])
            }
        }

        impl $trait<f64> for Node {
            type Output = Node;

            fn $method(self, other: f64) -> Self {
                Node::new($op, vec![Box::new(self), Box::new(c(other))])
            }
        }

        impl $trait<Node> for f64 {
            type Output = Node;

            fn $method(self, other: Node) -> Node {
                Node::new($op, vec![Box::new(c(self)), Box::new(other)])
            }
        }
    };
}

impl_op!(Add, add, Operation::Add);
impl_op!(Mul, mul, Operation::Mul);
