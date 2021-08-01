# Plain folding

### Dependencies

#### General

* Python 3
* PIL (pillow)
* NumPy (numpy)

#### Dataset visualization

* Matplotlib (matplotlib)
* CV2 (opencv-python)

#### Pix2Pix

* torch
* CV2 (opencv-python)

### Data examples*
<img alt="examples.png" src="examples.png" width=50% />

\*generated images with inverted colors and black contour 

### Train Pix2Pix model

generate folding dataset:

```
python gen_pix2pix_dataset.py
```

move unfolding dataset into pix2pix folder 
```
mv unfolding_1000_fc_3_af_0p2 pytorch-CycleGAN-and-pix2pix/datasets/unfolding_1000_fc_3_af_0p2/
```

start  model training:
```
cd  pytorch-CycleGAN-and-pix2pix
python train.py --dataroot ./datasets/unfolding_1000_fc_3_af_0p2 --name pix2pix_unfolding_1000_fc_3_af_0p2 --model pix2pix --direction BtoA
python train.py --dataroot ./datasets/unfolding_1000_fc_3_af_0p2 --name cycle_gan_unfolding_1000_fc_3_af_0p2 --model cycle_gan
```

eval on real img data:
```
python test_on_real_imgs.py --dataroot ./datasets/unfolding_1000_fc_3_af_0p2 --name pix2pix_unfolding_1000_fc_3_af_0p2 --model pix2pix 
```

### Train U-Net model

[U-Net training](unet/README.md)
