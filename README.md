#  Adversarial Sequence-to-sequence Domain adaptation 

## Overview
We propose a novel Adversarial Sequence-to-sequence Domain Adaptation Network dubbed ASSDA for robust text image recognition, 
which could adaptively transfer coarse global-level and  fine-grained character-level knowledge. 




### Install

1.  This code is test in the environment with ```cuda==10.1, python==3.6.8```.

2. Install Requirements

```
pip3 install torch==1.2.0 pillow==6.2.1 torchvision==0.4.0 lmdb nltk natsort
```

### Dataset

- The prepared synthetic and real scene dataset can be downloaded from [here](https://drive.google.com/drive/folders/192UfE9agQUMNq6AgU3_E05_FcPZK4hyt), which are created by  NAVER Corp. 

   - Synthetic scene text : [MJSynth (MJ)](http://www.robots.ox.ac.uk/~vgg/data/text/) and [SynthText (ST)](http://www.robots.ox.ac.uk/~vgg/data/scenetext/) \
   - Real scene text : the union of the training sets [IC13](http://rrc.cvc.uab.es/?ch=2), [IC15](http://rrc.cvc.uab.es/?ch=4), [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), and [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset).\
   - Benchmark evaluation scene text datasets : consist of [IIIT](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html), [SVT](http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset), [IC03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions), [IC13](http://rrc.cvc.uab.es/?ch=2)[3], [IC15](http://rrc.cvc.uab.es/?ch=4),
    [SVTP](http://openaccess.thecvf.com/content_iccv_2013/papers/Phan_Recognizing_Text_with_2013_ICCV_paper.pdf), and [CUTE](http://cs-chan.com/downloads_CUTE80_dataset.html).
- The prepared handwritten text dataset can be downloaded from [here](https://www.dropbox.com/sh/4a9vrtnshozu929/AAAZucKLtEAUDuOufIRDVPOTa?dl=0)    
    - Handwritten text: [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)


### Training and evaluation

- For a toy example, you can download the pretrained model from [here](https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW)
   
    - Add  model files to test into `data/`

- Training model
    
     ```
    CUDA_VISIBLE_DEVICES=1 python train_da_global_local_selected.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn \
     --src_train_data ./data/data_lmdb_release/training/ \
    --tar_train_data ./data/IAM/test --tar_select_data IAM --tar_batch_ratio 1 --valid_data ../data/IAM/test/ \
    --continue_model ./data/TPS-ResNet-BiLSTM-Attn.pth \
    --batch_size 128 --lr 1 \
    --experiment_name _adv_global_local_synth2iam_pc_0.1 --pc 0.1
    ```
    
- Test model

  - Test the baseline model
    ```
    CUDA_VISIBLE_DEVICES=0 python test.py   --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn   \
     --eval_data ./data/IAM/test \
     --saved_model ./data/TPS-ResNet-BiLSTM-Attn.pth 
    ```
    
  - Test the adaptation model
  
      ```
    CUDA_VISIBLE_DEVICES=0 python test.py   --Transformation TPS --FeatureExtraction ResNet --SequenceModeling BiLSTM --Prediction Attn   \
     --eval_data ./data/IAM/test \
     --saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Seed1111_adv_global_local_selected/best_accuracy.pth
    ```


##  Acknowledgement

This implementation has been based on this repository [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

