# ConvLSTM model for Weather Forecasting

PyTorch Lightning implementation of drought forecasting (classification) model (Convolutional LSTM). Classification is based on [PDSI index](https://en.wikipedia.org/wiki/Palmer_drought_index), and its corresponding bins. 

<img src="https://raw.githubusercontent.com/VGrabar/Weather-Prediction-NN/multiclass/docs/pdsi_bins.png" width="400" height="250">

We solve binary classification problem, where threshold for a drought could be adjusted in config file.

## Docker container launch

First, build an image

```
docker build . -t=<docker_image_name>
```
Then run a container with required parameters

```
docker run --mount type=bind,source=/local_path/Droughts/,destination=/Droughts/ -p <port_in>:<port_out> --memory=64g --cpuset-cpus="0-7" --gpus '"device=0"'  -it --rm --name=<docker_container_name>  <docker_image_name>
```

## Preprocessing ##

Input is geospatial monthly data, downloaded as .tif from public sources (e.g. from Google Earth Engine) and put into "data/raw" folder. Naming convention is "region_feature.tif". Please run

```
python3 preprocess.py --region region_name --band feature_name --endyear last_year_of_data --endmonth last_month_of_data
```

Results (both as .csv and .npy files) could be found in "data/preprocessed" folder.

## Training ##

To train model - first, change configs of datamodule and network (if necessary), edit necessary parameters (e.g. data path in train.yaml) - and then run
```
python3 train.py --config==train.yaml
```

Experiments results can be tracked via Comet ML (please add your token to logger config file or export it as enviromental variable)

## Inference ##

To run model on test dataset, calculate metrics and save predictions - first, change configs of datamodule and network (if necessary), add path to model checkpoint - and then run
```
python3 test.py --config==test.yaml

