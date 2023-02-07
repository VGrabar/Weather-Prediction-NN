import glob

import numpy as np
import re
import torch
import tqdm
from torchvision import transforms

from src.datamodules.weather_datamodule import Dataset_RNN
from src.models.rcnn_module import RCNNModule
from src.models.conv1d_module import Conv1dModule

dim_x = 40
dim_y = 20
hist_len = 96
per_forw = 1

reshape = transforms.Compose([transforms.Resize((dim_x, dim_y))])
celled_data_path = "data/celled/pdsi_voronezh.csv"
celled_data = torch.load(celled_data_path)
test_dataset = Dataset_RNN(
    celled_data,
    start_date=0,
    end_date=celled_data.shape[0],
    periods_forward=per_forw,
    history_length=hist_len,
    transforms=reshape,
)


chkpts0 = glob.glob("chkpts/pdsi_voronezh.csv_src.models.rcnn_module.RCNNModule_history_96_forward_36/**/*.ckpt", recursive=True)
chkpts0 = [c for c in chkpts0 if "last" not in c]

chkpts0 = ["convlstm1.ckpt"]
num_samples = len(chkpts0)
save_test_dataset = test_dataset.data[:-1]
print(save_test_dataset.shape)
np.save("voronezh_train", save_test_dataset)


forecast_horizon = 36
len_test = int(0.3*save_test_dataset.shape[0])

for j in range(-len_test-hist_len,-forecast_horizon-hist_len,1):
    if j == -len_test-hist_len:
        curr_test_dataset = save_test_dataset[j:j + hist_len]
        curr_test_dataset = curr_test_dataset[None, :, :, :]
        true_values = save_test_dataset[j + hist_len:j + hist_len + forecast_horizon]
        true_values = true_values[None, :, :, :]
    else:
        curr_sample = save_test_dataset[j:j + hist_len]
        curr_sample = curr_sample[None, :, :, :]

        curr_test_dataset = torch.cat((curr_test_dataset,  curr_sample), 0)
        curr_true = save_test_dataset[j + hist_len:j + hist_len + forecast_horizon]
        curr_true = curr_true[None, :, :, :]
        true_values = torch.cat((true_values,  curr_true), 0)

print("test", curr_test_dataset.shape)
np.save("voronezh_curr_test_dataset", curr_test_dataset)
#true_values = 
print("gtruth", true_values.shape)
np.save("voronezh_test_last_true", true_values)


len_test = curr_test_dataset.shape[0]
predictions = torch.zeros((num_samples, len_test, forecast_horizon, dim_x, dim_y))


i = 0
for curr_ckpt in chkpts0:
        
    device = torch.device('cpu')
    curr_model = RCNNModule.load_from_checkpoint(curr_ckpt)
    curr_model.batch_size = 1
    curr_model.register_buffer(
        "prev_state_h",
        torch.zeros(
            curr_model.batch_size,
            curr_model.hid_size,
            curr_model.n_cells_hor,
            curr_model.n_cells_ver,
            requires_grad=False,
        ),
    )
    curr_model.register_buffer(
        "prev_state_c",
        torch.zeros(
            curr_model.batch_size,
            curr_model.hid_size,
            curr_model.n_cells_hor,
            curr_model.n_cells_ver,
            requires_grad=False,
        ),
    )

    curr_preds = torch.zeros((len_test, forecast_horizon, dim_x, dim_y))

    for j in tqdm.tqdm(range(len_test)):
        sample = curr_test_dataset[j]
        sample = sample[None, :, :, :]
        for k in range(forecast_horizon):
            curr_preds[j][k] = curr_model(sample)
            sample[0][:-1] = sample[0][1:].clone()
            sample[0][-1] = curr_preds[j][k]


    predictions[i] = curr_preds
        
    i += 1

predictions = predictions.cpu().data.numpy()
print(predictions.shape)
np.save("voronezh_test_last_convlstm", predictions)
