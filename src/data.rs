use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};

pub struct CSVBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> CSVBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct CSVBatch<B: Backend> {
    pub features: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<Tensor<B, 2>, CSVBatch<B>> for CSVBatcher<B> {
    fn batch(&self, items: Vec<Tensor<B, 2>>) -> CSVBatch<B> {
        let features = items.iter().map(|item| item.clone()).collect();

        let targets = items
            .iter()
            .map(|item| {
                let mean = item.clone().mean().into_scalar();
                Tensor::<B, 1, Int>::from_data(Data::from([mean.elem()]))
            })
            .collect();

        let features = Tensor::cat(features, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        CSVBatch { features, targets }
    }
}
