# MinkLoc3D for nuScenes Radar Dataset
- This repo is modified from [`jac99/MinkLoc3D`](https://github.com/jac99/MinkLoc3D)

### Environment
* Ubuntu 18.04, CUDA 10.2, GeForce RTX 2070 Mobile / Max-Q Refresh
* Python 3.8.8
* PyTorch 1.8.1
* MinkowskiEngine 0.5.2 (note the version difference between the [`jac99/MinkLoc3D`](https://github.com/jac99/MinkLoc3D) and this repo results in an API change: ME.utils.sparse_quantize(coords>>>coordinates, feats>>>features))
* pytorch-metric-learning 0.9.98

### Original paper environment (ignore)
Code was tested using Python 3.8 with PyTorch 1.7 and MinkowskiEngine 0.4.3 on Ubuntu 18.04 with CUDA 10.2.

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

### nuScenes Radar Dataset Processing
Boston split has 17785 frames, devide it into four parts: database, train_query, val_query, test_query. 

- Train: stack database and train_query to form a mixed 'train tuple', where the length of query = len(database+trainquery).

- Val: database vs. val_query
 
```generate pickles
cd generating_queries/ 

# Generate training tuples for the Baseline Dataset
python generate_training_tuples_baseline.py

# Generate evaluation tuples
python generate_test_sets.py
```

```
├── MinkLoc3D_ws
│   ├── benchmark_datasets
│   ├── minkloc3d_marked
│   ├── minkloc3d_nuscenes
│   ├── nuscenes_radar
│   │   └── 7n5s_1024_xy1__drop
```

### Training
To train **MinkLoc3D** network, download and decompress the dataset and generate training pickles as described above.
Edit the configuration file (`config_baseline.txt`). 
Set `dataset_folder` parameter to the dataset root folder.
Modify `batch_size_limit` parameter depending on available GPU memory. 
Default limit (=256) requires at least 11GB of GPU RAM.

To train the network, run:

```train baseline
cd training

# To train minkloc3d model on the Baseline Dataset
python train.py --config ../config/config_baseline.txt --model_config ../models/minkloc3d.txt

```

### Evaluation

To evaluate pretrained models run the following commands:

```eval baseline
cd eval

# To evaluate the model trained on the Baseline Dataset
python evaluate.py --config ../config/config_baseline.txt --model_config ../models/minkloc3d.txt --weights ../weights/minkloc3d_baseline.pth
```

## Results

| Method         | nuScenes Radar  |
| -------------- |---------------- | 
| **MinkLoc3D**  |     **25.24**   |


