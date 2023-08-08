# Enhancement by Your Aesthetic: An Intelligible Unsupervised Personalized Enhancer for Low-Light Images (ACM MM2022)

[Paper link](https://arxiv.org/pdf/2207.07317.pdf)

## How to test on LOL

1. Update the paths of image sets and pre-trained models.
 ```
Updating the paths in configure files of /personalizedEnh/options/test/Enhancement/test_enhance_LOL.yml
```

2. Run the testing commands.
 ```
python test -opt /personalizedEnh/options/test/Enhancement/test_enhance_LOL.yml
```

## How to train iUP-Enhancer

**Some steps require replacing your local paths.**

1. Training the decomposition network.
```
python train.py -opt /personalizedEnh/options/train/Enhancement/train_decom.yml
```

2. Training the denoising network.
```
python train.py -opt /personalizedEnh/options/train/Enhancement/train_denoise.yml
```

3. Fine-tuning the denoising network.
```
python train.py -opt /personalizedEnh/options/train/Enhancement/train_denoiseFinetune.yml
```

4. Training the personalized enhancement network.
```
python train.py -opt /personalizedEnh/options/train/Enhancement/train_enhanceCondHis.yml
```
