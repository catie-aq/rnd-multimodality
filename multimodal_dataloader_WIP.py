import torch
import pandas as pd
from PIL import Image
import random

from vqvae import VQVAE

from torch.utils.data import Dataset, DataLoader
from torchvision import utils
import torchvision.transforms as transforms
from datetime import datetime, timedelta
from chronos import ChronosPipeline, ChronosConfig, ChronosTokenizer, MeanScaleUniformBins

class DatasetMultimodal(Dataset):
    def __init__(self, df_pandas, absolute_start, absolute_end, transforms, images_tokenizer, pipeline_ts):

        #get a list of random start&end dates
        L=[]        
        for i in range(1000):
            absolute_start = datetime.strptime(str(absolute_start), "%Y-%m-%d %H:%M:%S")
            absolute_end = datetime.strptime(str(absolute_end), "%Y-%m-%d %H:%M:%S")
            time_delta = absolute_end - absolute_start
            random_delta1 = random.uniform(0, time_delta.total_seconds())
            random_delta2 = random.uniform(0, time_delta.total_seconds())
            random_datetime1 = absolute_start + timedelta(seconds=random_delta1)
            random_datetime2 = absolute_start + timedelta(seconds=random_delta2)
            date_start, date_end = (random_datetime1, random_datetime2) if random_datetime1 < random_datetime2 else (random_datetime2, random_datetime1)
            L.append((date_start, date_end))

 
        self.seq_dates = L
        self.transforms = transforms
        self.images_tokenizer = images_tokenizer
        self.pipeline_ts = pipeline_ts
        self.df_pandas = df_pandas

    def __len__(self):
        return len(self.seq_dates)
    
    def __getitem__(self, idx):
        
        items_paths=[]
        while len(items_paths)==0: 
            date_start, date_end = self.seq_dates[idx]
            df_filtered = self.df_pandas.loc[(self.df_pandas['date'] >= str(date_start)) & (self.df_pandas['date'] <= str(date_end))]
            df_filtered.reset_index(drop=True, inplace=True)
            items_paths = df_filtered['file_path'].tolist()
            idx=idx+1

        items_paths=items_paths[:8092]

        n_tokens=0
        all_tokens_tensor = torch.empty(0)
        all_tokens_tensor = all_tokens_tensor.to("cuda")
        
        for i in range(len(items_paths)):
                        
            if df_filtered['type'][i]=="ts" :

                print(i, "one ts timestep used")
                for n_column in range(1,9):
                    one_value = df_filtered.iloc[0, n_column].astype(float) #dd
                    context = torch.tensor( one_value )
                    context = context.reshape(1,1)
                    new_token, _ = self.pipeline_ts.embed(context) # il faudra un positionnal encoding geographique par station + taille petite et pas 512.
                    new_token=new_token[:,1,:].to("cuda")
                    new_token= new_token.unsqueeze(dim=2)
                    all_tokens_tensor = torch.cat((all_tokens_tensor, new_token), dim=2)
                    n_tokens=n_tokens +1
                
            else :      
                print(i, "one image used")    
                image_path = df_filtered["file_path"][i]
                image = Image.open(image_path)
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image = self.transforms(image)

                image = image.to("cuda")
                self.images_tokenizer.to("cuda")
                self.images_tokenizer.eval()
                image = image.unsqueeze(dim=0) #model expects batches
                quant_t, quant_b, _, _, _= self.images_tokenizer.encode(image)
                
                quant_t_flat = torch.flatten(quant_t, start_dim=2)
                quant_b_flat = torch.flatten(quant_b, start_dim=2)
                new_tokens = torch.cat((quant_t_flat, quant_b_flat), dim=2)
                #this is ugly. pads from size 64 to 512 
                zeros_to_add = torch.zeros(1, 512 - 64, new_tokens.size(2)).to("cuda")
                new_tokens = torch.cat((new_tokens, zeros_to_add), dim=1)

                all_tokens_tensor = torch.cat((all_tokens_tensor, new_tokens), dim=2)
                n_tokens=n_tokens+320

            if all_tokens_tensor.size(2)>8092:
                break
        
        if all_tokens_tensor.size(2)>8092:
            all_tokens_tensor=all_tokens_tensor[:,:,:8092] 
        return  all_tokens_tensor, n_tokens
   
"""
def collate_fn(data): # masquage aleatoire, extraction du contenu caché, pad a la taille max
    
    sequence, mask_sequence = zip(*data)
    #faire le dynamic padding et renvoyer batch_masqué, contenu des trous
    tokens_concat, lengths = zip(*data)

    #les masquer aléatoirement et sortir le contenu des trous
    #padding + batch

    max_len = max(lengths)
    n_ftrs = data[0][0].size(1)
    features = torch.zeros((len(data), max_len, n_ftrs))
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)

    for i in range(len(data)):
        j, k = data[i][0].size(0), data[i][0].size(1)
        features[i] = torch.cat([data[i][0], torch.zeros((max_len - j, k))])

    return batch_masqué, batch_des_labels
"""

if __name__ == "__main__":

    transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
    
    images_tokenizer = VQVAE()
    images_tokenizer.load_state_dict(torch.load("vqvae_005.pt"))

    pipeline_ts = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",
    device_map="cuda",
    torch_dtype=torch.bfloat16,
    )
    
    df_pandas= df = pd.read_csv("fused_data_ts_images.csv")
    absolute_start = "2016-01-01 00:00:00"
    absolute_end =  "2019-01-01 00:00:00"
    
    dataset_images = DatasetMultimodal(df_pandas, absolute_start, absolute_end, transforms, images_tokenizer, pipeline_ts)
    print("Longueur totale du dataset:", len(dataset_images))
    
    seq_tokens, n = dataset_images[0]
    print("taille seq tokens", seq_tokens.size() )

    dataloader_images = DataLoader(dataset_images, shuffle=True, batch_size=10 )
    print("Taille des lots:", dataloader_images.batch_size)
    for batch_idx, out in enumerate(dataloader_images):
        print("Batch", batch_idx)
        print("tokens:", out[0].size())  

        if batch_idx >= 1:
            break

        
    """#check all by recontructing one image

    to do : tenir compte de la dimension 512 à rammener à 64 pour les images

    tokens=out[0][0,:,:,:320] #first tokens of the previous batch
    quant_t, quant_b = tokens[:,:,:64],tokens[:,:,64:]
    quant_t=quant_t.reshape(1,64,8,8)
    quant_b=quant_b.reshape(1,64,16,16)
    output = images_tokenizer.decode( quant_t, quant_b )
    utils.save_image( output, "verif_reconstruction.png", normalize=True, range=(-1,1))
    print("fin")"""


