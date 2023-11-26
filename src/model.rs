use burn::autodiff::ADBackendDecorator;
use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::WgpuBackend;
use burn::config::Config;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLoss;
use burn::nn::Dropout;
use burn::nn::DropoutConfig;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::nn::ReLU;
use burn::tensor::backend::ADBackend;
use burn::tensor::backend::Backend;
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};

pub type MyBackend = WgpuBackend<AutoGraphicsApi, f32, i32>;
pub type MyAutodiffBackend = ADBackendDecorator<MyBackend>;

use crate::data::CSVBatch;

#[derive(Config, Debug)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self) -> Model<B> {
        Model {
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }

    /// Returns the initialized model using the recorded weights.
    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init_with(record.linear1),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes)
                .init_with(record.linear2),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
    activation: ReLU,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, features: Tensor<B, 2>) -> Tensor<B, 1> {
        let batch_size = features.dims()[0];

        let x = features.reshape([batch_size, 16 * 8 * 8]);

        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = x.flatten(1, 1);
        self.linear2.forward(x)
    }

    pub fn forward_classification(
        &self,
        features: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(features);
        let output = output.unsqueeze(); // Reshape the output tensor to Tensor<B, 2>
        let loss = CrossEntropyLoss::new(None).forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: ADBackend> TrainStep<CSVBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: CSVBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.features, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<CSVBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: CSVBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.features, batch.targets)
    }
}
