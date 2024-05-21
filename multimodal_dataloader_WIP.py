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
from torch.nn.utils.rnn import pad_sequence

class DatasetMultimodal(Dataset):
    def __init__(self, df_pandas, absolute_start, absolute_end, transforms, images_tokenizer, pipeline_ts):

        #get a list of random start&end dates
        L=[]        
        for i in range(4):
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
        print("getitem is called")
        items_paths=[]
        while len(items_paths)==0: 
            date_start, date_end = self.seq_dates[idx]
            df_filtered = self.df_pandas.loc[(self.df_pandas['date'] >= str(date_start)) & (self.df_pandas['date'] <= str(date_end))]
            df_filtered.reset_index(drop=True, inplace=True)
            items_paths = df_filtered['file_path'].tolist()
            idx=idx+1

        items_paths=items_paths[:8192]

        n_tokens=0
        all_tokens_tensor = torch.empty(0)
        all_tokens_tensor = all_tokens_tensor.to("cuda")
        
        for i in range(len(items_paths)):
                        
            if df_filtered['type'][i]=="ts" :

                print(i, "one ts timestep used")
                for n_column in range(1,9): #on parcourt chaque variable indépendamment
                    one_value = df_filtered.iloc[0, n_column].astype(float) 
                    context = torch.tensor( one_value )
                    context = context.reshape(1,1) 
                    new_token, _ = self.pipeline_ts.embed(context) # il faudra un positionnal encoding geographique par station, VARIABLE + taille petite et pas 512.
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
                zeros_to_add = torch.zeros(1, 512 - 64, new_tokens.size(2)).to("cuda") #this is ugly. pads from size 64 to 512
                new_tokens = torch.cat((new_tokens, zeros_to_add), dim=1)

                all_tokens_tensor = torch.cat((all_tokens_tensor, new_tokens), dim=2)
                n_tokens=n_tokens+320

            if all_tokens_tensor.size(2)>8192:
                break
        
        if all_tokens_tensor.size(2)>8192:
            all_tokens_tensor=all_tokens_tensor[:,:,:8192] 
            n_tokens = 8192
        return  all_tokens_tensor, n_tokens
   

def my_collate_fn(batch):
    print("collat_fn is called")
    # Create lists to hold the masked sequences and the original contents of the masked positions
    masked_batch = []
    batch_of_labels = []

    print("aaaaaaaaaaa", len(batch) )
    print("bbbbbbbbbbbbbbb", type(batch[0]))

    for sequence in batch[0]:
        sequence_length = len(sequence)
        mask_indices = random.sample(range(sequence_length), k=int(0.2 * sequence_length))
        
        masked_sequence = sequence.clone()
        labels = torch.zeros_like(sequence)

        for idx in mask_indices:
            labels[idx] = sequence[idx]
            masked_sequence[idx] = 0  # Assuming 0 is the masking value

        masked_batch.append(masked_sequence)
        batch_of_labels.append(labels)

    # Pad sequences to the max length
    masked_batch = pad_sequence(masked_batch, batch_first=True, padding_value=0)
    batch_of_labels = pad_sequence(batch_of_labels, batch_first=True, padding_value=0)

    return masked_batch, batch_of_labels


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
    
    dataset_multi = DatasetMultimodal(df_pandas, absolute_start, absolute_end, transforms, images_tokenizer, pipeline_ts)
    print("Longueur totale du dataset multimodal(nombre de séquences différentes):", len(dataset_multi))
           
           
    #contrôle sur une sequence
    """print("génère une seuqnce")
    seq_tokens, n = dataset_multi[0]
    print("taille de le premiere sequence", seq_tokens.size() )"""

    dataloader_multi = DataLoader(dataset_multi, shuffle=True, batch_size=1, collate_fn = my_collate_fn)
    print("Taille des lots:", dataloader_multi.batch_size)

    for batch_idx, out in enumerate(dataloader_multi):
        print("Batch", batch_idx)
        print("contenu du batch, tokens", out[0].size())  
        print("contenu du batch, partie n_tokens:", out[1]) 

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
