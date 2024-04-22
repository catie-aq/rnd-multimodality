import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
from chronos import ChronosPipeline, ChronosTokenizer, MeanScaleUniformBins, ChronosConfig


class MultimodalDataset(Dataset):
    def __init__(self, df, target, time_length):
        self.ts= df[target]
        self.time_length = time_length
        # self.config = ChronosConfig(**config_args)

    def __len__(self):
        return len(self.ts) // self.time_length

    def __getitem__(self, idx):
      
        ts = self.ts[idx*self.time_length:(idx+1)*self.time_length]
        
        return torch.tensor(ts.tolist())



if __name__ == "__main__":
    path = "/mnt/data1/Datasets/Multimodal_v2/ts-data/french_weather/"
    df = pd.read_csv(path + "every6m-data2018.csv")
    metoe_dataset = MultimodalDataset(df,"t",10)
    print(metoe_dataset[0])
    meteo_dataloader = DataLoader(metoe_dataset, batch_size=4, shuffle=True)
    # load model
    pipeline_tiny = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    config_args = pipeline_tiny.model.model.config.chronos_config
    config = ChronosConfig(**config_args)
    tokenizer = config.create_tokenizer()
    count =0
    for context in meteo_dataloader:
        token_ids, attention_mask, scale = tokenizer.input_transform(context)
        print(token_ids)
        count = count +1
        if count ==3:
            break