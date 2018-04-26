# Relational-Networks
Pytorch implementation of " A simple neural network module for relational reasoning" paper aka Relational networks for visual reasoning. https://arxiv.org/abs/1706.01427

<img src='https://writelatex.s3.amazonaws.com/rtdjcknkvwxj/uploads/806/23070795/1.png?X-Amz-Expires=14400&X-Amz-Date=20180426T141042Z&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJF667VKUK4OW3LCA/20180426/us-east-1/s3/aws4_request&X-Amz-SignedHeaders=host&X-Amz-Signature=f4fc0388dfee0ebc2fb88b46611348a18ae561aa620e2a24ca43949e20d14f93' />

## Important Note

This implementation includes only the visual pipeline for CLEVR dataset. Best validation accuracy acheived with this implementation is **72%** compared to **96.8%** reported in the paper. This result was acheived by applying a learning rate schedule that doubles the learning rate every 20 epochs (motivated by warmup in https://arxiv.org/abs/1706.02677). The paper itself does not discuss any schedules used, running with schedules gets **65%** at best.

Pull requests and suggestions are welcome to reproduce the results from the paper.

## Training and Valiation Accuracies with warmup


<p float="left">
  <img src='https://user-images.githubusercontent.com/8495451/39027855-db99cb10-4421-11e8-9303-9d1b93e1389a.png' width='420' height='420'/>
  <img src='https://user-images.githubusercontent.com/8495451/39027856-dbacaa8c-4421-11e8-8903-7dbcf8f26be5.png' width='420' height='420' />
</p>

## Requirements

- [Pytorch](http://pytorch.org/)
- [Visdom](https://github.com/facebookresearch/visdom)
- [H5py](https://www.h5py.org/)
- [opencv-python](https://anaconda.org/conda-forge/opencv)


## Usage

### Train

`python3 runtime`

**Arguments**

- `lr` : Learning rate. default: `2.5e-4`
- `batch_size`: default : `64`
- `warmup`: A flag to turn on doubling the learning rate every 20 epochs. default: `False`
- `save_path`: path to checkpoints. Checkpoints are saved for every new best validation accuracy.
- `vis_screen`: Visdom env name. default: `RelNet`

## Other Implementations (Visual pipeline)

- https://github.com/rosinality/relation-networks-pytorch
- https://github.com/mesnico/RelationNetworks-CLEVR
