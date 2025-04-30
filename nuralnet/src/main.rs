use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::distributions::Uniform;

// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Derivative of the sigmoid function
fn sigmoid_derivative(output: f64) -> f64 {
    output * (1.0 - output)
}

#[derive(Debug)]
pub struct NeuralNetwork {
    input_neurons: usize,
    hidden_neurons: usize,
    output_neurons: usize,
    learning_rate: f64,
    epochs: usize,
    weights_input_hidden: Array2<f64>,
    weights_hidden_output: Array2<f64>,
    bias_hidden: Array1<f64>,
    bias_output: Array1<f64>,
}

impl NeuralNetwork {
    pub fn new(input_neurons: usize, hidden_neurons: usize, output_neurons: usize, learning_rate: f64, epochs: usize) -> Self {

        // initialising wieghts
        let weights_input_hidden = Array2::random((input_neurons, hidden_neurons), Uniform::new(-1.0, 1.0));
        let weights_hidden_output = Array2::random((hidden_neurons, output_neurons), Uniform::new(-1.0, 1.0));

        // initialising bias 
        let bias_hidden = Array1::random(hidden_neurons, Uniform::new(-1.0, 1.0));
        let bias_output = Array1::random(output_neurons, Uniform::new(-1.0, 1.0));

        NeuralNetwork {
            input_neurons,
            hidden_neurons,
            output_neurons,
            learning_rate,
            epochs,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
        }
    }

    pub fn forward(&self, input: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        // calculating hidden layer -> input*wieghts + bias and applying sigmoid activation to the outut
        let hidden_layer_input = input.dot(&self.weights_input_hidden) + &self.bias_hidden;
        let hidden_layer_output = hidden_layer_input.mapv(sigmoid);
        // calculating final layer -> input*wieghts + bias and applying sigmoid activation to the outut
        let final_layer_input = hidden_layer_output.dot(&self.weights_hidden_output) + &self.bias_output;
        let final_output = final_layer_input.mapv(sigmoid);

        (hidden_layer_output, final_output)
    }

    pub fn backward(&mut self, input: &Array2<f64>, target: &Array2<f64>, output: &Array2<f64>, hidden_output: &Array2<f64>) -> f64 {
        // error calculation for output
        let error = target - output;
        let d_output = error.clone() * output.mapv(sigmoid_derivative);
        // error calculation for hidden
        let error_hidden = d_output.dot(&self.weights_hidden_output.t());
        let d_hidden = error_hidden.clone() * hidden_output.mapv(sigmoid_derivative);

        // Calculate the change in weights for the hidden-to-output layer
        let delta_weights_hidden_output = hidden_output.t().dot(&d_output) * self.learning_rate;
        self.weights_hidden_output = &self.weights_hidden_output + &delta_weights_hidden_output;

        // Calculate the change in bias for the hidden-to-output layer
        let delta_bias_output = d_output.sum_axis(Axis(0)) * self.learning_rate;
        self.bias_output = &self.bias_output + &delta_bias_output;

        // Calculate the change in weights for the input-to-hidden layer
        let delta_weights_input_hidden = input.t().dot(&d_hidden) * self.learning_rate;
        self.weights_input_hidden = &self.weights_input_hidden + &delta_weights_input_hidden;

        // Calculate the change in biases for the hidden layer
        let delta_bias_hidden = d_hidden.sum_axis(Axis(0)) * self.learning_rate;
        self.bias_hidden = &self.bias_hidden + &delta_bias_hidden;

        error.mapv(f64::abs).sum() / (error.len() as f64)
    }

    pub fn train(&mut self, input: &Array2<f64>, target: &Array2<f64>) {
        for epoch in 0..self.epochs {
            let (hidden_output, output) = self.forward(input);
            let loss = self.backward(input, target, &output, &hidden_output);
            if epoch % 1000 == 0 {
                println!("Epoch {}, Loss: {:.4}", epoch, loss);
            }
        }
    }

    pub fn predict(&self, input: &Array2<f64>) -> Array2<f64> {
        // Perform the forward pass to get the output probabilities
        let (_, output) = self.forward(input);

        // Convert the probabilities to binary predictions (>= 0.5 becomes 1.0, < 0.5 becomes 0.0)
        output.mapv(|val| if val >= 0.5 { 1.0 } else { 0.0 })
    }

}

fn main() {
    // Training data for the XOR problem
    // Input: [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    // Target Output: [[0.0], [1.0], [1.0], [0.0]]
    let x: Array2<f64> = Array::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    let y: Array2<f64> = Array::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0]).unwrap();

    // Create a new neural network
    // 2 input neurons (for the two input features)
    // 2 hidden neurons (a common choice for XOR)
    // 1 output neuron (for the single output)
    // Learning rate of 0.5
    // Train for 10000 epochs
    let mut nn = NeuralNetwork::new(2, 2, 1, 0.5, 10000);

    // Train the neural network
    nn.train(&x, &y);

    // Make predictions after training
    println!("Final Output:");
    println!("{:?}", nn.predict(&x));
}
