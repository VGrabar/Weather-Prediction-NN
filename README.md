# ConvLSTM model for Weather Forecasting

PyTorch Lightning implementation of drought classification model, partly inspired by the paper ["Recurrent Convolutional Neural Networks help to predict location of Earthquakes"](https://arxiv.org/abs/2004.09140) and its Convolutional LSTM model. Classification is based on [PDSI index](https://en.wikipedia.org/wiki/Palmer_drought_index), and its corresponding bins. 

![](https://raw.githubusercontent.com/VGrabar/Weather-Prediction-NN/multiclass/docs/pdsi_bins.png).

We are solving both binary and multi-class tasks, which could be altered by changing `boundaries` value in config file.
Input is geospatial data (e.g. pre-processed files from ERA5 or TerraClimate), which should be uploaded as .csv file (with date, value, x_dim and y_dim columns).  

To train model - first, change configs of datamodule and network (if necessary) - and then run
```
python train.py --config==train.yaml
```

To test pre-trained model - first, change configs of datamodule and network (if necessary) - and then run (with a correct path to checkpoint)
```
python test.py --config==test.yaml
```

Results of experiments can be tracked via Comet ML 

