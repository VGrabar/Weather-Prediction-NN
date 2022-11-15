import glob

import numpy as np
import re
import torch
from torchvision import transforms

from datamodules.weather_datamodule import Dataset_RNN
from models.rcnn_module import RCNNModule

dim_x = 40
dim_y = 80
reshape = transforms.Compose([transforms.Resize((dim_x, dim_y))])
celled_data_path = "data/celled/pdsi_voronezh.csv"
celled_data = torch.load(celled_data_path)
test_dataset = Dataset_RNN(
    celled_data,
    start_date=0,
    end_date=celled_data.shape[0],
    periods_forward=0,
    history_length=12,
    transforms=reshape,
)

range_forward = 36
num_samples = 21
predictions = torch.zeros((range_forward, num_samples, dim_x, dim_y))

chkpts = glob.glob("/path/to/*.ckpt", recursive=True)
for f in range(1, range_forward):
    
    regex = r'forward=+{periods}.*'.format(periods=f)
    i = 0
    for curr_ckpt in chkpts:
        if re.search(regex, curr_ckpt):
            curr_model = RCNNModule.load_from_checkpoint()
            curr_preds = curr_model(test_dataset[-1][0])
            predictions[f - 1, i, :, :] = curr_preds
            i += 1

predictions = predictions.cpu().data.numpy()
np.save("voronezh_predictions.csv", predictions)
