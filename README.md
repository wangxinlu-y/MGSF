# Empowering Vision Foundation Models via Multi-Granularity Selective Fine-tuning for Remote Sensing Scene Classification of Coastal Zones
## CROSS-14 Dataset
For coastal remote sensing scene classification.

![Visio Diagram](Fig1.png)

the dataset is avaliable at：https://pan.baidu.com/s/1FSQz4Dyi-gO0PzV-bu0E9A?pwd=tue1

The link of Google griver: https://drive.google.com/file/d/1tQFQQXOCkb-2iaF0NhB9u_O45db04HkD/view?usp=sharing

## How to run
1. Run `train_linear.py` for linear probing.  
2. Run `modify_weights.py` to compute uncertainty-aware sample weights.  
3. Run `train_atten.py` to train with token- and channel-granularity selection.
