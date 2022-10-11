# Object Relation Transformer

<img width="1341" alt="Screen Shot 2022-10-11 at 19 55 33" src="https://user-images.githubusercontent.com/51681991/195072462-d8a07e64-2b96-4ac0-a952-554134801f26.png">

- [Image Captioning: Transforming Objects into Words](https://arxiv.org/abs/1906.05963)
- This repository is based on [the repo](https://github.com/yahoo/object_relation_transformer).
- This code uses [STAIR Captions](http://captions.stair.center/) for training Japanese image captioning model.
  - So you should download STAIR Captions and [COCO datasets](https://cocodataset.org/#download)


## Requirements
* Python 3.8.10
* PyTorch

Run:

```
pip install -r requirements.txt
```

## Data Preparation

### Download ResNet101 weights for feature extraction

Download the file `resnet101.pth` from [here](https://drive.google.com/drive/folders/0B7fNdx_jAqhtbVYzOURMdDNHSGM). Copy the weights to a folder `imagenet_weights` within the data folder:

```
mkdir data/imagenet_weights
cp /path/to/downloaded/weights/resnet101.pth data/imagenet_weights
```

### preprocess the COCO captions

<!--- Download the [preprocessed COCO captions](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `stair_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.
-->

```
$ python scripts/prepro_stair_labels.py
```

Then run:

```
$ python scripts/prepro_labels.py --input_json data/stair_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```
`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

Next run (python 2.7 is recommend):
```
$ python2 scripts/prepro_ngrams.py --input_json data/stair_coco.json --dict_json data/cocotalk.json --output_pkl data/coco-train --split train
```

This will preprocess the dataset and get the cache for calculating cider score.


### Download the COCO dataset and pre-extract the image features

Download the [COCO images](http://mscoco.org/dataset/#download) from the MSCOCO website.
We need 2014 training images and 2014 validation images. You should put the `train2014/` and `val2014/` folders in the same directory, denoted as `$IMAGE_ROOT`:

```
mkdir $IMAGE_ROOT
pushd $IMAGE_ROOT
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
popd
wget https://msvocds.blob.core.windows.net/images/262993_z.jpg
mv 262993_z.jpg $IMAGE_ROOT/train2014/COCO_train2014_000000167126.jpg
```

The last two commands are needed to address an issue with a corrupted image in the MSCOCO dataset (see [here](https://github.com/karpathy/neuraltalk2/issues/4)). The prepro script will fail otherwise.


Then run:

```
$ python scripts/prepro_feats.py --input_json data/stair_coco.json --output_dir data/cocotalk --images_root $IMAGE_ROOT
```

`prepro_feats.py` extracts the ResNet101 features (both fc feature and last conv feature) of each image. The features are saved in `data/cocotalk_fc` and `data/cocotalk_att`, and resulting files are about 200GB. Running this script may take a day or more, depending on hardware.

(Check the prepro scripts for more options, like other ResNet models or other attention sizes.)

### Download the Bottom-up features

Download the pre-extracted features from [here](https://github.com/peteanderson80/bottom-up-attention). For the paper, the adaptive features were used.

Do the following:
```
mkdir data/bu_data; cd data/bu_data
azcopy copy https://imagecaption.blob.core.windows.net/imagecaption/trainval.zip
unzip trainval.zip
```
(It is recommended to download large files with AzCopy for faster speed. AzCopy executable tools can be downloaded [here](https://learn.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10#download-azcopy).)

The .zip file is around 22 GB.
Then return to the base directory and run:
```
python scripts/make_bu_data.py --output_dir data/cocobu
```

This will create `data/cocobu_fc`, `data/cocobu_att` and `data/cocobu_box`.


### Generate the relative bounding box coordinates for the Relation Transformer

Run the following:
```
python scripts/prepro_bbox_relative_coords.py --input_json data/stair_coco.json --input_box_dir data/cocobu_box --output_dir data/cocobu_box_relative --image_root $IMAGE_ROOT
```
This should take a couple hours or so, depending on hardware.


## Model Training and Evaluation

### Standard cross-entropy loss training

```
python train.py --id relation_transformer_bu --caption_model relation_transformer --input_json data/cocotalk.json --input_fc_dir data/cocobu_fc --input_att_dir data/cocobu_att --input_box_dir data/cocobu_box --input_rel_box_dir data/cocobu_box_relative --input_label_h5 data/cocotalk_label.h5 --checkpoint_path log_relation_transformer_bu --noamopt --noamopt_warmup 10000 --label_smoothing 0.0 --batch_size 15 --learning_rate 5e-4 --num_layers 6 --input_encoding_size 512 --rnn_size 2048 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --save_checkpoint_every 6000 --language_eval 1 --val_images_use 5000 --max_epochs 30 --use_box 1
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `save/`). We only save the best-performing checkpoint on validation and the latest checkpoint to save disk space.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

If you have tensorflow, the loss histories are automatically dumped into `--checkpoint_path`, and can be visualized using tensorboard.

The current command uses scheduled sampling. You can also set scheduled_sampling_start to -1 to disable it.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the [coco-caption code](https://github.com/tylin/coco-caption) into `coco-caption` directory.

For more options, see `opts.py`.


The above training script should achieve a CIDEr-D score of about 115.

## License

This project is licensed under the terms of the MIT open source license. Please refer to [LICENSE](LICENSE) for the full terms.

```
@article{herdade2019image,
  title={Image Captioning: Transforming Objects into Words},
  author={Herdade, Simao and Kappeler, Armin and Boakye, Kofi and Soares, Joao},
  journal={arXiv preprint arXiv:1906.05963},
  year={2019}
}
```
