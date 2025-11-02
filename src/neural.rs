use crate::rvec::{self, AsRVec as _, RVec, rvec};
use flux_rs::assert;
use flux_rs::attrs::*;
use rand::{Rng, rngs::ThreadRng};
use crate::range::{spread_i32, spread_usize};

fn test() {
    assert(10 < 20)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// NOTE: We don't refine f64 due to a bug in the implementation.
#[vars(
    $wk0(n, m) = [n == m];
    $wk1(n, m) = [true];
)]
#[spec(fn(&RVec<f64>[@n], &RVec<f64>[@m]) -> f64
       requires $wk0(n, m)
       ensures  $wk1(n, m)
)]
fn dot_product(a: &RVec<f64>, b: &RVec<f64>) -> f64 {
    let mut sum = 0.0;
    for i in spread_usize(0, a.len()) {
        sum += a[i] * b[i];
    }
    sum
}

#[vars(
    $wk0(n, m) = [n == m];
    $wk1(n, m) = [true];
)]
#[spec(fn(&RVec<f64>[@n], &RVec<f64>[@m]) -> f64
       requires $wk0(n, m)
       ensures  $wk1(n, m)
)]
fn dot_product2(a: &RVec<f64>, b: &RVec<f64>) -> f64 {
    spread_usize(0,a.len()).map_f64(|i| (a[i] * b[i])).sum()
}

// HT: https://byteblog.medium.com/building-a-simple-neural-network-from-scratch-in-rust-3a7b12ed30a9

// Define the structure of a single layer in the network
#[refined_by(i: int, o: int)]
struct Layer {
    #[field(usize[i])]
    num_inputs: usize,

    #[field(usize[o])]
    num_outputs: usize,

    #[field(RVec<RVec<f64>[i]>[o])]
    weight: RVec<RVec<f64>>,

    #[field(RVec<f64>[o])]
    bias: RVec<f64>,

    #[field(RVec<f64>[o])]
    outputs: RVec<f64>,
}

#[vars(
    $wk0(n) = [true];
    $wk1(v, n) = [0 <= v, v < n];
    $wk2(v, n) = [v == n];
)]
#[spec(fn(n: usize, f:F) -> RVec<A>[#v]
       requires $wk0(n)
       ensures $wk2(v, n)
       where F: FnMut(usize{v: $wk1(v, n)}) -> A
)]
fn init0<F, A>(n: usize, mut f: F) -> RVec<A>
where
    F: FnMut(usize) -> A,
{
    let mut i = 0;
    let mut res = RVec::new();
    while i < n {
        res.push(f(i));
        i += 1;
    }
    res
}

#[vars(
    $wk0(n) = [true];
    $wk1(v, n) = [0 <= v, v < n];
    $wk2(v, n) = [v == n];
)]
#[spec(fn(n: usize, f:F) -> RVec<T>[#v]
       requires $wk0(n)
       ensures $wk2(v, n)
       where F: FnMut(usize{v: $wk1(v, n)}) -> T
)]
fn init<T, F>(n: usize, mut f: F) -> RVec<T>
where
    F: FnMut(usize) -> T,
{
    let mut res = RVec::new();
    for i in spread_usize(0, n) {
        res.push(f(i));
    }
    res
}

// #[vars(
//     $wk0(n) = [];
//     $wk1(v, n) = [0 <= v, v < n];
//     $wk2(v, n) = [v == n];
// )]
// #[spec(fn(n: usize, f:F) -> RVec<f64>[#v]
//        requires $wk0(n)
//        ensures $wk2(v, n)
//        where F: FnMut(usize{v: $wk1(v, n)}) -> f64
// )]
// fn init_rvec<F>(n: usize, mut f: F) -> RVec<f64>
// where
//     F: FnMut(usize) -> f64,
// {
//     let mut res = RVec::new();
//     for i in spread_usize(0, n) {
//         res.push(f(i));
//     }
//     res
// }
// 
// #[vars(
//     $wk0(n) = [];
//     $wk1(v, n) = [0 <= v, v < n];
//     $wk2(v, n) = [v == n];
// )]
// #[spec(fn(n: usize, f:F) -> RVec<RVec<f64>>[#v]
//        requires $wk0(n)
//        ensures $wk2(v, n)
//        where F: FnMut(usize{v: $wk1(v, n)}) -> RVec<f64>
// )]
// fn init_rvec_rvec<F>(n: usize, mut f: F) -> RVec<RVec<f64>>
// where
//     F: FnMut(usize) -> RVec<f64>,
// {
//     let mut res = RVec::new();
//     for i in spread_usize(0, n) {
//         res.push(f(i));
//     }
//     res
// }

#[vars(
    $wk0(n) = [true];
    $wk1(v, n) = [0 <= v, v < n];
    $wk2(v, n) = [v == n];
)]
#[spec(fn(n: usize, f:F) -> RVec<RVec<f64>>[#v]
       requires $wk0(n)
       ensures $wk2(v, n)
       where F: FnMut(usize{v: $wk1(v, n)}) -> RVec<f64>
)]
fn init2<F>(n: usize, mut f: F) -> RVec<RVec<f64>>
where
    F: FnMut(usize) -> RVec<f64>,
{
    spread_usize(0, n).map_rvec_f64(|i| f(i)).collect()
}

#[vars(
    $wk0(input_size, output_size) = [true];
    $wk1(v, input_size, output_size) = [v == output_size];
    $wk2(inner, v, input_size, output_size) = [inner == input_size];
)]
#[spec(fn(input_size: usize, output_size: usize) -> RVec<RVec<f64>{inner: $wk2(inner, v, input_size, output_size)}>[#v]
       requires $wk0(input_size, output_size)
       ensures $wk1(v, input_size, output_size)
)]
fn mk_weights(input_size: usize, output_size: usize) -> RVec<RVec<f64>> {
    let mut rng = rand::thread_rng();
    let weights = init(output_size, |_| {
        // replaced `rng.gen_range(-1.0..1.0)` with `0.0`
        init(input_size, |_| 0.0)
    });
    weights
}

impl Layer {
    #[vars(
        $wk0(i, o) = [true];
        $wk1(l, i, o) = [l == Layer{ i : i, o : o }];
    )]
    #[spec(fn(i: usize, o: usize) -> Layer[#l]
           requires $wk0(i, o)
           ensures  $wk1(l, i, o)
    )]
    fn new(i: usize, o: usize) -> Layer {
        let mut rng = rand::thread_rng();
        Layer {
            num_inputs: i,
            num_outputs: o,
            // replaced `rng.gen_range(-1.0..1.0)` with `0.0`
            weight: init(o, |_| init(i, |_| 0.0)),
            // replaced `rng.gen_range(-1.0..1.0)` with `0.0`
            bias: init(o, |_| 0.0),
            outputs: init(o, |_| 0.0),
        }
    }

    #[vars(
        $wk0(l, n) = [n == l.i];
        $wk1(l, n) = [true];
    )]
    #[spec(fn(&mut Layer[@l], &RVec<f64>[@n])
           requires $wk0(l, n)
           ensures  $wk1(l, n)
    )]
    fn forward(&mut self, input: &RVec<f64>) {
        spread_usize(0, self.num_outputs).for_each(|i| {
            let weighted_input = dot_product(&self.weight[i], input);
            self.outputs[i] = sigmoid(weighted_input + self.bias[i])
        })
    }

    #[vars(
        $wk0(l, m, n) = [m == l.i, n == l.o];
        $wk1(v, l, m, n) = [v == l.i];
    )]
    #[spec(fn(&mut Layer[@l], &RVec<f64>[@m], &RVec<f64>[@n], _) -> RVec<f64>[#v]
           requires $wk0(l, m, n)
           ensures  $wk1(v, l, m, n)
    )]
    fn backward(&mut self, inputs: &RVec<f64>, error: &RVec<f64>, learning_rate: f64) -> RVec<f64> {
        let mut input_error = rvec![0.0; inputs.len()];
        for i in spread_usize(0,self.num_outputs) {
            for j in spread_usize(0, self.num_inputs) {
                input_error[j] += self.weight[i][j] * error[i];
                self.weight[i][j] -= learning_rate * error[i] * inputs[j];
            }
            self.bias[i] -= learning_rate * error[i];
        }
        input_error
    }
}

#[vars(
    $wk0(n, m) = [n == m];
    $wk1(n, m) = [true];
)]
#[spec(fn(&RVec<f64>[@n], &RVec<f64>[@m]) -> f64
       requires $wk0(n, m)
       ensures  $wk1(n, m)
)]
fn mean_squared_error(predicted: &RVec<f64>, actual: &RVec<f64>) -> f64 {
    spread_usize(0, predicted.len())
        .map_f64(|i| (predicted[i] - actual[i]).powi(2))
        .sum::<f64>()
        / predicted.len() as f64
}

// -------------------------------------------------------------------------------------

#[refined_by(i: int, o: int)]
enum NeuralNetwork {
    #[variant((Layer[@i, @o]) -> NeuralNetwork[i, o])]
    Last(Layer),

    #[variant((Layer[@i, @h], Box<NeuralNetwork[h, @o]>) -> NeuralNetwork[i, o])]
    Next(Layer, Box<NeuralNetwork>),
}

impl NeuralNetwork {
    /// Create a new neural network with the given input size, hidden layer sizes, and output size.
    #[vars(
        $wk0(i, o) = [true];
        $wk1(n, i, o) = [n == NeuralNetwork{ i : i, o : o }];
    )]
    #[spec(fn(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> NeuralNetwork[#n]
           requires $wk0(input_size, output_size)
           ensures  $wk1(n, input_size, output_size)
    )]
    fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> NeuralNetwork {
        if hidden_sizes.len() == 0 {
            NeuralNetwork::Last(Layer::new(input_size, output_size))
        } else {
            let n = hidden_sizes[0];
            let rest = NeuralNetwork::new(n, &hidden_sizes[1..], output_size);
            let layer = Layer::new(input_size, n);
            NeuralNetwork::Next(layer, Box::new(rest))
        }
    }

    #[vars(
        $wk0(i, o, n) = [n == i];
        $wk1(v, i, o, n) = [v == o];
    )]
    #[spec(fn(&mut NeuralNetwork[@i, @o], &RVec<f64>[@n]) -> RVec<f64>[#v]
           requires $wk0(i, o, n)
           ensures  $wk1(v, i, o, n)
    )]
    fn forward(&mut self, input: &RVec<f64>) -> RVec<f64> {
        match self {
            NeuralNetwork::Last(layer) => {
                layer.forward(input);
                layer.outputs.clone()
            }
            NeuralNetwork::Next(layer, next) => {
                layer.forward(input);
                next.forward(&layer.outputs)
            }
        }
    }

    /// Backpropagation algorithm: assumes we have already done a "forwards" pass with
    /// the results stored in each `Layer`'s `outputs` field.
    #[vars(
        $wk0(i, o, n, m) = [n == i, m == o];
        $wk1(v, i, o, n, m) = [v == i];
    )]
    #[spec(fn(&mut NeuralNetwork[@i, @o], &RVec<f64>[@n], &RVec<f64>[@m], _) -> RVec<f64>[#v]
           requires $wk0(i, o, n, m)
           ensures  $wk1(v, i, o, n, m)
    )]
    fn backward(
        &mut self,
        inputs: &RVec<f64>,
        target: &RVec<f64>,
        learning_rate: f64,
    ) -> RVec<f64> {
        match self {
            NeuralNetwork::Last(layer) => {
                let error = spread_usize(0, layer.num_outputs)
                    .map_f64(|i| layer.outputs[i] - target[i])
                    .collect();
                layer.backward(inputs, &error, learning_rate)
            }
            NeuralNetwork::Next(layer, next) => {
                let error = next.backward(&layer.outputs, target, learning_rate);
                layer.backward(inputs, &error, learning_rate)
            }
        }
    }
}
