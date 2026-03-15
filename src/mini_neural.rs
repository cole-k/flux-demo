use crate::range::{spread_i32, spread_usize};

fn random() -> f64 {
    0.0
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/*
  INPUT
       ←---------- n ----------→
       [ x₁,  x₂,  x₃, ..., xₙ ]
       [ y₁,  y₂,  y₃, ..., yₙ ]

  OUTPUT
       ( x₁   x₂   x₃       xₙ )
   SUM (  ·    ·    ·  ...   · )
       ( y₁,  y₂,  y₃,    , yₙ )
*/
fn dot_product(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    let mut i = 0;
    let mut sum = 0.0;
    while i < x.len() {
        sum += x[i] * y[i];
        i += 1;
    }
    sum
}

/*
  INPUT
       n
       f() -> T

  OUTPUT
       ←-------- n -------→
       [f(), f(), ..., f()]
*/
fn init<T, F>(n: usize, mut f: F) -> Vec<T>
where
    F: FnMut() -> T,
{
    let mut res = Vec::new();
    for _ in spread_usize(0, n) {
        res.push(f());
    }
    res
}
/*
  INPUT
      input_size  (n)
      output_size (m)

  OUTPUT
      ←---------- n -----------→
      ┌    r₁₁  r₁₂  ... r₁ₙ   ┐
      |    r₂₁  r₂₂  ... r₂ₙ   | ↑
      |     :    :        :    | m
      └    rm₁  rm₂  ... rmₙ   ┘
*/
fn mk_weights(input_size: usize, output_size: usize) -> Vec<Vec<f64>> {
    init(output_size, || init(input_size, || random()))
}

/*
             Weights (m × n)           Inputs (n)           Output (m)
       ┌                         ┐    ┌────────┐
       │ w11  w12  w13  ...  w1n │      input1        ┌                       ┐
       │ w21  w22  w23  ...  w2n │  ×   input2     =  │ out1  out2  ...  outm │
       │  .    .    .         .  │        .           └                       ┘
       │  .    .    .         .  │      inputn
       │  .    .    .         .  │    └────────┘
       │ wm1  wm2  wm3  ...  wmn │
       └                         ┘
*/
fn forward(weights: &Vec<Vec<f64>>, inputs: &Vec<f64>) -> Vec<f64> {
    let mut outputs = Vec::new();
    for row in spread_usize(0, weights.len()) {
        let weighted_input = dot_product(&weights[row], inputs);
        outputs.push(sigmoid(weighted_input));
    }
    outputs
}

fn run(inputs: &Vec<f64>, output_len: usize) -> Vec<f64> {
    let weights = mk_weights(inputs.len(), output_len);
    forward(&weights, inputs)
}
