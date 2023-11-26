use crate::data::CSVBatcher;
use crate::model::ModelConfig;
use crate::training::TrainingConfig;
use anyhow::Result;
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use rand::seq::SliceRandom;
use std::error::Error;
use std::path::Path;

pub mod data;
pub mod model;
pub mod training;

fn main() {
    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "./tmp";
    crate::training::train::<crate::model::MyAutodiffBackend>(
        artifact_dir,
        crate::training::TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device,
    );
}

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: Tensor<B, 2>) {
    let config =
        TrainingConfig::load(format!("{artifact_dir}/config.json")).expect("A config exists");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into())
        .expect("Failed to load trained model");

    let model = config.model.init_with::<B>(record).to_device(&device);

    let batcher = CSVBatcher::<B>::new(device.clone());
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.features);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();
    let label = batch.targets.into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}

pub fn split_csv_into_train_test(
    input_path: &Path,
    train_path: &Path,
    test_path: &Path,
    test_ratio: f32,
) -> Result<(), Box<dyn Error>> {
    // Read the input CSV file
    let mut rdr = csv::Reader::from_path(input_path)?;
    let headers = rdr.headers()?.clone();
    let mut records: Vec<csv::StringRecord> = rdr.records().collect::<Result<_, _>>()?;

    // Shuffle the records
    let mut rng = rand::thread_rng();
    records.shuffle(&mut rng);

    // Split the records into training and test sets
    let test_size = (records.len() as f32 * test_ratio).round() as usize;
    let (test_records, train_records) = records.split_at(test_size);

    // Write the training set
    let mut wtr = csv::Writer::from_path(train_path)?;
    wtr.write_record(&headers)?;
    for record in train_records {
        wtr.write_record(record)?;
    }
    wtr.flush()?;

    // Write the test set
    let mut wtr = csv::Writer::from_path(test_path)?;
    wtr.write_record(&headers)?;
    for record in test_records {
        wtr.write_record(record)?;
    }
    wtr.flush()?;

    Ok(())
}
