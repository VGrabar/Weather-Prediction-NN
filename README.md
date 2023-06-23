# ConvLSTM model for Weather Forecasting

PyTorch Lightning implementation of drought classification model, partly inspired by the paper ["Recurrent Convolutional Neural Networks help to predict location of Earthquakes"](https://arxiv.org/abs/2004.09140) and its Convolutional LSTM model. Classification is based on [PDSI index](https://en.wikipedia.org/wiki/Palmer_drought_index), and its corresponding bins. 

<img src="https://raw.githubusercontent.com/VGrabar/Weather-Prediction-NN/multiclass/docs/pdsi_bins.png" width="400" height="250">

We are solving both binary and multi-classification tasks, which could be altered by changing `boundaries` value in config file - for example, setting them to `[-2, 2]` in order to solve 3-class problem.
Input is geospatial data (e.g. pre-processed files from ERA5 or TerraClimate), which should be uploaded as .csv file (with _date_, _value_, _x_dim_ and _y_dim_ columns).  

To train model - first, change configs of datamodule and network (if necessary) - and then run
```
python train.py --config==train.yaml
```

To test pre-trained model - first, change configs of datamodule and network (if necessary) - and then run (with a correct path to checkpoint in _test.yaml_)
```
python test.py --config==test.yaml
```

Results of experiments can be tracked via Comet ML 

