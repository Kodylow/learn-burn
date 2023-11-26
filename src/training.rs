use std::fs::File;
use std::io::BufReader;

use crate::{data::CSVBatcher, model::ModelConfig};
use burn::data::dataset::InMemDataset;
use burn::module::Module;
use burn::tensor::Data;
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{backend::ADBackend, Tensor},
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};
use csv::ReaderBuilder;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B: ADBackend<FloatElem = f32>>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: B::Device,
) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Save without error");

    B::seed(config.seed);

    // Load training data
    let train_file = File::open("./data/train.csv").expect("Could not open train.csv");
    let mut train_reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(train_file));
    let train_dataset: Vec<Vec<f32>> = train_reader
        .deserialize()
        .collect::<Result<_, _>>()
        .expect("Could not deserialize train.csv");

    // Load testing data
    let test_file = File::open("./data/test.csv").expect("Could not open test.csv");
    let mut test_reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(BufReader::new(test_file));
    let test_dataset: Vec<Vec<f32>> = test_reader
        .deserialize()
        .collect::<Result<_, _>>()
        .expect("Could not deserialize test.csv");

    let train_num_cols = train_dataset[0].len();
    let test_num_cols = test_dataset[0].len();

    let train_data: Vec<Tensor<B, 2>> = train_dataset
        .into_iter()
        .map(|row| {
            let data = Data::from(row.as_slice());
            Tensor::from_data(data).reshape([1, train_num_cols])
        })
        .collect();

    let batcher_train = CSVBatcher::<B>::new(device.clone());
    let batcher_valid = CSVBatcher::<B::InnerBackend>::new(device.clone());

    let train_dataset: InMemDataset<Tensor<B, 2>> = InMemDataset::new(train_data);
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let test_data: Vec<Tensor<B::InnerBackend, 2>> = test_dataset
        .into_iter()
        .map(|row| {
            let data = Data::from(row.as_slice());
            Tensor::from_data(data).reshape([1, test_num_cols])
        })
        .collect();

    let test_dataset: InMemDataset<Tensor<B::InnerBackend, 2>> = InMemDataset::new(test_data);
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");
}
