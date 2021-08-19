# MinkLoc3D for nuScenes Radar Dataset
This repo is modified from [`jac99/MinkLoc3D`](https://github.com/jac99/MinkLoc3D).

### MinkLoc3D paper environment
Code was tested with Python 3.8 with PyTorch 1.7 and MinkowskiEngine 0.4.3 on Ubuntu 18.04 with CUDA 10.2.

The following Python packages are required:
* PyTorch (version 1.7)
* MinkowskiEngine (version 0.4.3)
* pytorch_metric_learning (version 0.9.94 or above)
* tensorboard
* pandas
* psutil
* bitarray

Modify the `PYTHONPATH` environment variable to include absolute path to the project root folder: 
```export PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/.../MinkLoc3D
```

### milliPlace paper environment
* Ubuntu 18.04, CUDA 10.2, GeForce RTX 2070 Mobile / Max-Q Refresh
* Python 3.8.8
* PyTorch 1.8.1
* MinkowskiEngine 0.5.2 (note the version discrepancy between the [`jac99/MinkLoc3D`](https://github.com/jac99/MinkLoc3D) and this repo results in an API change: ME.utils.sparse_quantize(coords > change to > coordinates, feats > change to > features))
* pytorch-metric-learning 0.9.98




### nuScenes dataset pre-processing
Boston split has 17785 frames, which are divided into four splits: `database`, `train_query`, `val_query`, `test_query`. 

- train phase: stack `database` and `train_query` to form a mixed 'train tuple', where the length of query = len(`database`+`train_query`).

- val phase: `database` vs. `val_query`

- test phase: `database` vs. `val_query`
 
copy the processed nuScenes dataset (from milliPlace) to the following directory:
```
├── minkloc3d_milliPlace
│   ├── nuscenes_radar
│   │   └── 7n5s_xy11
```

generate pickles
```
cd minkloc3d_milliPlace/nuscenes_dataset/ 
./generate.sh
```

### Training

Edit the configuration file `config_baseline.txt`:
- `dataset_folder` : the dataset root folder.
- `batch_size_limit` : depends on available GPU memory (default limit (256) requires at least 11GB of GPU RAM).

Start training:

```
cd minkloc3d_milliPlace

python training/train.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt

```

### Evaluation

```
cd minkloc3d_milliPlace

python eval/evaluate.py --config ./config/config_baseline.txt --model_config ./models/minkloc3d.txt --weights ./weights/model_MinkFPN_GeM_20210819_1446_final.pth
```

### Results

MinkLoc3D on nuScenes radar dataset: Recall@1/5/10 = 31.8% / 53.6% / 61.1%.


