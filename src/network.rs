use std::ops::Add;

use ndarray::{Array, Array2, ArrayBase, OwnedRepr, Axis};
use rand::{thread_rng, Rng};

#[derive(Default)]
pub struct NetworkBuilder {
    learning_rate: Option<f64>,
    input_size: Option<usize>,
    output_size: Option<usize>,
    hidden: Vec<LayerBuilder>,
}

impl NetworkBuilder {

    #[inline]
    pub fn new() -> Self {
        Default::default()
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }

    pub fn output_size(mut self, output_size: usize) -> Self {
        self.output_size = Some(output_size);
        self
    }

    pub fn hidden(mut self, layer: LayerBuilder) -> Self {
        self.hidden.push(layer);
        self
    }

    pub fn build(self) -> Network {
        let input_size = self.input_size.expect("there was no input given");
        let output_size = self.output_size.expect("there was no output size given");
        let mut layers = Vec::with_capacity(self.hidden.len() + 1);
        
        for layer in 0..self.hidden.len() {
            let neurons = self.hidden[layer].neurons.expect("neurons need to be defined inside hidden layers");
            let layer = Layer::new(layers.last().map(|layer: &Layer| layer.neurons()).unwrap_or(input_size), neurons, false); // FIXME: why is a type annotation needed here?
            layers.push(layer);
        }

        layers.push(Layer::new(layers.last().expect("there has to be at least 1 hidden layer").neurons(), output_size, true));
        
        Network {
            learning_rate: self.learning_rate.expect("there was no learning rare given"),
            input_size,
            layers,
        }
    }

}

pub struct LayerBuilder {
    neurons: Option<usize>,
}

impl LayerBuilder {

    pub const fn new() -> Self {
        Self {
            neurons: None,
        }
    }

    pub const fn neurons(mut self, neurons: usize) -> Self {
        self.neurons = Some(neurons);
        self
    }

}

pub struct Network {
    learning_rate: f64,
    input_size: usize,
    layers: Vec<Layer>,
}

impl Network {

    pub fn eval(&self, input: Vec<f64>) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size);
        
        self.forward_prop(input).last().unwrap().post_activation
    }

    fn forward_prop(&self, input: Vec<f64>) -> Vec<LayerEvalResult> {
        let mut results = vec![];
        let mut input = Array::from_shape_vec((1, self.input_size), input).unwrap();
        for layer in self.layers.iter() {
            let weights = Array2::<f64>::from_shape_vec((layer.neurons(), layer.inputs()), layer.weights.clone()).unwrap();

            let result = weights.dot(&input);
            let mut result = result.add(Array::from_shape_vec((10, 1), layer.biases.clone()).unwrap());
            result.map_inplace(|e| {
                *e = e.exp();
            });
            let mut result = result.sum_axis(Axis(0));
            /*let sum = result.sum();
            result /= sum;*/

            let raw_pre = result.to_vec();


            if !layer.output {
                // apply relu function
                result.map_inplace(|e| {
                    *e = e.max(0.0);
                });
            }


            results.push(LayerEvalResult {
                pre_activation: raw_pre,
                post_activation: result.to_vec(),
            });

            input = result.into_shape((1, self.input_size)).unwrap();
            /*for neuron in 0..layer.neurons() {
                let weights = layer.weights[neuron];
                
                let val = layer.biases[neuron] + VecStorage::new(weights.len(), 1, weights);
            }*/
        }
        results
    }

    pub fn train(&self, input: Vec<f64>, result: Vec<f64>) -> Vec<f64> {
        let forward_prop = self.forward_prop(input);
        
    }

}

struct LayerEvalResult {
    pre_activation: Vec<f64>,
    post_activation: Vec<f64>,
}

pub struct Layer {
    weights: Vec<f64>,
    biases: Vec<f64>,
    output: bool,
}

impl Layer {

    fn new(inputs: usize, neurons: usize, output: bool) -> Self {
        let mut rng = thread_rng();
        Self {
            weights: {
                let mut weights = Vec::with_capacity(neurons);
                for _ in 0..neurons {
                    for _ in 0..inputs {
                        weights.push(rng.gen_range(-0.5..0.5));
                    }
                }
                weights
            },
            biases: {
                let mut biases = Vec::with_capacity(neurons);
                for _ in 0..neurons {
                    biases.push(rng.gen_range(-0.5..0.5));
                }
                biases
            },
            output,
        }
    }

    #[inline]
    fn neurons(&self) -> usize {
        self.biases.len()
    }

    #[inline]
    fn inputs(&self) -> usize {
        self.weights.len() / self.neurons()
    }

}

/*#[inline]
fn relu(val: &mut ArrayBase<OwnedRepr<f64>, >) {
    for elem in val.as_mut_slice() {
        *elem = elem.max(0.0);
    }
}*/

#[inline]
fn deriv_relu(val: f64) -> f64 {
    (val > 0.0) as usize as f64
}

// softmax
