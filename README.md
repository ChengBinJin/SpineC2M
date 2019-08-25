# DC2Anet: Spine CT to MR (SpineC2M) Image Synthesis
This repository is a TensorFlow implementation of the paper "[DC2Anet: Generating Lumbar Spine MR Images from CT Scan Data Based on Semi-Supervised Learning](https://www.mdpi.com/2076-3417/9/12/2521)," Applied Sciences, 2019, 9(12), 2512.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/63273562-2a9af180-c2d9-11e9-851e-1102dfbe39c1.png")
</p>

## Requirements
- tensorflow 1.14.0
- opencv 4.1.0.25
- numpy 1.16.2
- matplotlib 3.0.3
- scipy 1.2.1
- pillow 5.4.1
- pickleshare 0.7.5
- xlsxwriter 1.1.5
- scikit-image 0.14.2
- scikit-learn 0.20.3
- xlrd 1.2.0

## CT-Based Synthetic MRI Generation Results
<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/63317667-aa16d800-c34e-11e9-963f-76ef124e56ea.png" width=1000)
</p>  

## Implementations
### Related Works
- [Multi-Channel GAN](https://arxiv.org/pdf/1707.09747.pdf)
- [Deep MR-to-CT](https://arxiv.org/pdf/1708.01155.pdf)
- [DiscoGAN](https://arxiv.org/pdf/1703.05192.pdf)
- [MR-GAN](https://www.mdpi.com/1424-8220/19/10/2361)
- [DC2Anet(Ours)](https://www.mdpi.com/2076-3417/9/12/2521)

### Objective Function
The objective function of DC2Anet contains six loss terms in total: adversarial, dual cycle-consistent, voxel-wise, gradient difference, perceptual, and structural similarity. A summary of the strengths and weaknesses of each loss term is given in bellow table.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/63356044-1969e780-c3a2-11e9-86a5-9c54934d95ca.png" width=800)
</p>  

### Hybrid Discriminator
The variant architectures of the hybrid discriminator. Models A-F are hybrid discriminators for the aligned and unaligned data flow. Model G consists of two independent discriminators.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/63488390-baaf8580-c4e9-11e9-8c34-7be29307b0fe.png" width=800)
</p>  

## Documentation
### Directory Hierarchy
``` 
SpineC2M
├── src
│   ├── cyclegan
│   │   └── ...
│   ├── DC2Anet
│   │   ├── dataset.py
│   │   ├── dc2anet.py
│   │   ├── main.py
│   │   ├── reader.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   ├── utils.py
│   │   └── vgg16.py
│   ├── discogan
│   │   └── ...
│   ├── eval
│   │   └── src
│   │   │   ├── ablationStudy.py
│   │   │   ├── case_bar.py
│   │   │   ├── cherry_pick.py
│   │   │   ├── csv_boxplot.py
│   │   │   ├── eval.py
│   │   │   ├── eval_iniv.py
│   │   │   ├── utils.py
│   │   │   └── write_csv.py
│   ├── mrgan
│   │   └── ...
│   ├── mrganPlus
│   │   └── ...
│   └── pix2pix
│   │   └── ...
Data
└── spine06
│   └── tfrecords
│   │   ├── train.tfrecords
│   │   └── test.tfrecords
Models_zoo
└── caffe_layers_value.pickle
```  

### Training DC2Anet
Under the folder of the DC2Anet, using `main.py` train a DC2Anet model. Example usage:
```
python main.py 
```
- `gpu_index`: gpu index if you have multiple gpus, default: `0`
- `is_cycle_consistent`: cycle-consistent loss for the generator, default: `True`
- `cycle_consistent_weight`: weight for the cycle-consistent loss term, default: `10.`
- `is_voxel`: voxel-wise loss for the generator, default: `True`
- `L1_lambda`: L1 lambda for conditional voxel-wise loss, default: `100.`
- `is_gdl`: gradient difference loss (GDL) for the generator, default: `True`
- `gdl_weight`: weight (hyper-parameter) for gradient difference loss term, default: `100.`
- `is_perceptual`: perceputal loss for the generator, default: `True`
- `perceptual_weight`: weight (hyper-parameter) for perceptual loss term, default: `1.`
- `perceptual_mode`: feature layers [1|2|3|4|5], default: `5`
- `is_ssim`: SSIM loss for the generator, default: `True`
- `ssim_weight`: weight (hyper-parameter) for ssim loss term, default: `0.05`
- `dis_model`: discriminator model, select from [a|b|c|d|e|f|g], default: `a`
- `learning_mode`: learning mode, select from [super, unsuper, semi], default: `semi`
- `is_alternative_optim`: optimizing by alterative or integrated optimization, default: `True`
- `is_lsgan`: use LSGAN loss, default: `False`
- `is_train`: training or inference mode, default: `True`
- `batch_size`: batch size for one iteration, default: `1`  
- `dataset`: dataset name, default: `spine06`  
- `learning_rate`: initial learning rate for Adam, default: `2e-4` 
- `beta1`: momentum term of Adam, default: `0.5`
- `iters`: number of iterations, default: `200000`  
- `print_freq`: print frequency for loss, default: `100`  
- `save_freq`: save frequency for model, default: `10000`
- `sample_freq`: sample frequency for saving image, default: `500`
- `load_model`: folder of saved model that you wish to continue training, (e.g. 20181127-2116), default: `None`  

### Test DC2Anet
Use `main.py` to test the DC2Anet model. Example usage:
```
python main.py --load_model=folder/you/wish/to/test/e.g./20181127-2116
```
please refer to the above arguments.

### Tensorboard Visualization
Tensorboard collects all of the loss terms. Under the folder `DC2Anet\spline06\logs`, using the following command:
```
tensorboard --logdir=folder/you/wish/to/test/e.g./20181127-2116
```

### Evaluation
All of the evaluation figures in the paper were generated by functions in the `eval/src` file.
- ablationStudy.py
- case_bar.py
- cherry_pick.py
- csv_boxplot.py
- eval.py
- eval_indv.py
- utils.py
- write_csv.py

### Ablation Study Results
Our objective function contained six independent loss terms. The experiments reported above used all of the loss terms. To investigate the strength of each loss term, we employed ablation analysis to determine how performance was affected by each loss term. We trained each network with a different objective funtion five times using different initialization weights and report the average of the five trials for each objective function. The evaluation results are shown in Table. In addition, the synthesis results for the ablation analysis are presented in the following figure.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/63496434-e12aec00-c4fc-11e9-8029-dbea22052232.png" width=800)
</p> 

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/63488397-c1d69380-c4e9-11e9-9f87-e1a8b0f58f8a.png" width=1000)
</p> 
  
### Comparison with Baselines
To compare synthesis MR images produced using different methods quantitatively, we present box plots in following figure representing the MAE, RMSE, PSNR, SSIM, and PCCs resulting from the use of [multi-channel GAN](https://arxiv.org/pdf/1707.09747.pdf), [deep MR-to-CT](https://arxiv.org/pdf/1708.01155.pdf), [DiscoGAN](https://arxiv.org/pdf/1703.05192.pdf), [MR-GAN](https://www.mdpi.com/1424-8220/19/10/2361), and our proposed method.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/63496784-74fcb800-c4fd-11e9-90b5-c97fcc0553bb.png" width=1000)
</p> 
  
### Citation
```
  @misc{chengbinjin2019DCNN,
    author = {Cheng-Bin Jin},
    title = {SpineC2M-Tensorflow},
    year = {2019},
    howpublished = {\url{https://github.com/ChengBinJin/SpineC2M},
    note = {commit xxxxxxx}
  }
```

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
