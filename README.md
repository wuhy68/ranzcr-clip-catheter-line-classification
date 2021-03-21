## RANZCR CLiP - Catheter and Line Position Challenge Silver Medal Solution

competition: https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification

### Data preparation

1. Download the  to `./data` using the scripts at https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/data

### Training

We use DDP AMP Training, run the following scripts:

```bash
./distributed_train.sh 4
```

After training, models and tranning logs will be saved in `./model/` by default.


### Predicting

This competition was a code competition. Teams submitted inference notebooks which were ran on hidden test sets. We made public the submission notebook on Kaggle at https://www.kaggle.com/haoyuanwu/ranzcrinference-howell

All the trained models are linked in that notebook as public datasets. The same notebook is also included in this repo for reference.

