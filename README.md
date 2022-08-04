# LSTM - CNN model for Weather Forecasting

PyTorch Lightning implementation of weather forecasting model, partly inspired by the paper ["Recurrent Convolutional Neural Networks help to predict location of Earthquakes"](https://arxiv.org/abs/2004.09140).  

Input is geospatial data, which could be uploaded as .csv file (with date, value, x_dim and y_dim columns).  

To train model - first, change configs of datamodule and network (if necessary) - and then run
```
python train.py --config==train.yaml
```

