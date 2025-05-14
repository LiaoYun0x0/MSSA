import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
from MSSA import ImageEncoder, Bert, MatchingModel
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tqdm import tqdm,trange
import random
import torch
import torch.nn as nn
import numpy as np
import numpy
import yaml
import data           
import argparse
import torch.nn.functional as F
from matplotlib import pyplot as plt
from ipdb import set_trace
import copy

from torch.utils.data.sampler import SubsetRandomSampler
# from tda import set_requires_grad
from transformers import BertTokenizer
import torch.optim as optim
from transformers import BertModel, BertConfig
from transformers import AutoConfig, AutoModel, AutoTokenizer
import sklearn.preprocessing
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial
from eval import ratk,i2t5,t2i5        
import csv
import json
import time 
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast  
from torch.utils.tensorboard import SummaryWriter
from config import parser_options                  
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')         
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import logging
import argparse
import cv2

logging.basicConfig(filename='model_processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def triplet_loss(emb_v, 
               emb_text_pos, 
               emb_text_neg, 
               emb_text, 
               emb_v_pos, 
               emb_v_neg,
               device,
              ):
    margin = 0.5     
    alpha = 1
 
    v_loss_pos = 2-torch.cosine_similarity(emb_v, emb_text_pos,dim=1) 
    v_loss_neg = 2-torch.cosine_similarity(emb_v, emb_text_neg,dim=1) 


    t_loss_pos = 2-torch.cosine_similarity(emb_text, emb_v_pos,dim=1) 
    t_loss_neg = 2-torch.cosine_similarity(emb_text, emb_v_neg,dim=1) 

  
    image_text_loss = torch.sum(torch.max(torch.zeros(1).to(device), margin + alpha * v_loss_pos - v_loss_neg))
    

    text_image_loss = torch.sum(torch.max(torch.zeros(1).to(device), margin + alpha * t_loss_pos - t_loss_neg))

    triplet_loss = (image_text_loss +text_image_loss)*0.1
    return triplet_loss, image_text_loss, text_image_loss


def main(args_re):
    torch.set_num_threads(1)        
    options = parser_options()       


    train_dataloader, _, test_dataloader = data.get_loaders(options["dataset"]["batch_size"], options)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   
 
    image_encoder = ImageEncoder(args_re).to(device)    
    text_encoder = Bert(args_re.bert_name).to(device)    
    model = MatchingModel(image_encoder,text_encoder,embed_dim=768, num_heads=8,drop_prob=0.1).to(device)


    optimizer = optim.AdamW(model.parameters(), lr=args_re.lr,
                             weight_decay=0.01, betas=(0.9, 0.999), eps=1.0e-8)
    
    mr = 0              
    ep = 1
    nums = 0
    total_loss = []  
    metric = []

    for epoch in range(1, args_re.epochs + 1):

        epoch_total_loss = 0
        total_image_text_loss = 0
        total_text_image_loss = 0
        # train_loss = 0
        nums = 0
        gradient_norms = [] 

        image_encoder.train()            
        text_encoder.train()
        model.train()

 
        for step, (rs_img, text,iids) in tqdm(enumerate(train_dataloader),total=len(train_dataloader),
                                              desc=f"Epoch [{epoch}/{args_re.epochs}]",leave=True):

            rs_img = rs_img.to(torch.float32).to(device)   
            token_input_ids = []    
            token_attentions = []      
            cap_len = []           
         
            for i in range(len(text)):
                # set_trace()
                '''text-model对文本进行tokenize '''
                token_ids = text_encoder.tokenizers.encode_plus(text[i],
                                                padding="max_length",  
                                                max_length=31,         
                                                add_special_tokens=True, 
                                                return_tensors='pt',    
                                                return_attention_mask=True,  
                                                truncation=True     
                                                )
                token_input_ids.append(token_ids['input_ids'][0])     
                token_attentions.append(token_ids['attention_mask'][0])   
                cap_len.append(int(token_ids['attention_mask'][0].sum().cpu().numpy()))  
          
        
            cap_len = np.array(cap_len)
            token_ids = torch.stack(token_input_ids).to(device)
            token_attentions = torch.stack(token_attentions).to(device)

            # set_trace()
            optimizer.zero_grad()                 
            set_requires_grad(model)    
#------------------------------------------------------------------------
            enhanced_image_features, enhanced_text_features = model(token_ids, token_attentions, rs_img,num_seed=token_attentions.shape[0])
          
            enhanced_image_features = enhanced_image_features.reshape(enhanced_image_features.size(0), -1)
            enhanced_text_features = enhanced_text_features.reshape(enhanced_text_features.size(0), -1)
    
            text_feature = enhanced_text_features
            rs_image_feature = enhanced_image_features
#------------------------------------------------------------------------   
            # triplet loss        
            adj_mat = np.eye(rs_img.shape[0])
            mask_mat_ = np.ones_like(adj_mat) - adj_mat            

            mask_mat = 1000000*adj_mat+mask_mat_    
        

            sim_it=scipy.spatial.distance.cdist(rs_image_feature.detach().cpu().numpy(), text_feature.detach().cpu().numpy(), 'cosine')
            img_sim_mat = mask_mat*sim_it

            img_neg_text_idx = np.argmin(img_sim_mat, axis=1).astype(int)  
            img_pos_text_idx = np.argmax(img_sim_mat, axis=1).astype(int)  

            img_neg_text = text_feature[img_neg_text_idx, :]   
            img_pos_text = text_feature  
            emb_t_neg = img_neg_text
            
            sim_ti = scipy.spatial.distance.cdist(text_feature.detach().cpu().numpy(), rs_image_feature.detach().cpu().numpy(), 'cosine')
            text_sim_mat = mask_mat*sim_ti
            text_neg_img_idx = np.argmin(text_sim_mat, axis=1).astype(int)
            text_pos_img_idx = np.argmax(text_sim_mat, axis=1).astype(int)
           
            text_neg_img = rs_image_feature[text_neg_img_idx, :]   
            text_pos_img = rs_image_feature  
            emb_v_neg = text_neg_img

            emb_v_pos = text_pos_img
            emb_t_pos = img_pos_text
 
            tripletloss, image_text_loss, text_image_loss= triplet_loss(rs_image_feature, 
            emb_t_pos, 
            emb_t_neg, 
            text_feature, 
            emb_v_pos, 
            emb_v_neg,
            device
            )

            loss = tripletloss
             

            loss.backward()


            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_total_loss += loss.item()
            total_image_text_loss += image_text_loss.item()
            total_text_image_loss += text_image_loss.item()
            nums += 1
        

        mean_loss = epoch_total_loss / nums
        total_loss.append(mean_loss)
        mean_image_text_loss = total_image_text_loss / nums
        mean_text_image_loss = total_text_image_loss / nums
        logging.info(f'Epoch [{epoch}/{args_re.epochs}], Mean Triplet Loss: {mean_loss:.4f}, Mean Image-Text Loss: {mean_image_text_loss:.4f}, Mean Text-Image Loss: {mean_text_image_loss:.4f}')
        print(f'Epoch [{epoch}/{args_re.epochs}], Loss: {mean_loss:.4f}')
        
        del rs_image_feature  #Delete variables to save  memory
        del text_feature
#-------------------------------------------------------------------------------
        #test   
        with torch.no_grad():
            image_encoder.eval()
            text_encoder.eval()
            model.eval()      

            image_features = []
            text_features = []
            image_paths = []
            original_texts = []

            for step, (rs_img, text,img_path) in tqdm(enumerate(test_dataloader),total=len(test_dataloader),
                                        desc=f"Epoch [{epoch}/{args_re.epochs}]",leave=True):
                

                start_time = time.time()
                rs_img = rs_img.to(torch.float32).to(device) 
                token_ids = text_encoder.tokenizers.encode_plus(text[0],
                                                    padding="max_length",
                                                    max_length=31,
                                                    add_special_tokens=True,
                                                    return_tensors='pt',
                                                    return_attention_mask=True,
                                                    truncation=True
                                                    )
                enhanced_image_features, enhanced_text_features = model(token_ids['input_ids'].to(device), token_ids['attention_mask'].to(device), rs_img,rs_img)
                enhanced_image_features = enhanced_image_features.reshape(enhanced_image_features.size(0), -1)
                enhanced_text_features = enhanced_text_features.reshape(enhanced_text_features.size(0), -1)
                text_feature = enhanced_text_features
                rs_image_feature = enhanced_image_features

                image_features.append(rs_image_feature[0])
 
                text_features.append(text_feature[0])
                image_paths.append(img_path[0])
                original_texts.append(text[0])  # 保留原始文本数据

            image_features = torch.stack(image_features).to(device)
            text_features = torch.stack(text_features).to(device)
            text_features = text_features
            image_features = image_features
          
            image_features = image_features.cpu().numpy().copy()
            text_features = text_features.cpu().numpy().copy()
            image_features = np.array([image_features[i] for i in range(0, len(image_features), 5)])

            t_r1, t_r5, t_r10,_,_ = i2t5(image_features,text_features, original_texts)   #image-text
            v_r1, v_r5, v_r10,_,_ = t2i5(image_features,text_features)   #text-image

            mean_rat = (t_r1+t_r5+t_r10+v_r1+v_r5+v_r10)/6     
            #更新最佳模型
            if mr <= mean_rat:
                mr = mean_rat
                ep = epoch
                tr1 = t_r1
                tr5 = t_r5
                tr10 = t_r10
                vr1 = v_r1
                vr5 = v_r5
                vr10 = v_r10

            metric.append(mean_rat)
            print(f"this epoch: {epoch}, mean_rat: {mean_rat},best mean_rat: {mr}, best epoch: {ep}")

            
            logging.info(f't_r1={t_r1}, t_r5={t_r5}, t_r10={t_r10}, '
                       f'v_r1={v_r1}, v_r5={v_r5}, v_r10={v_r10}, '
                       f'mr={(t_r1+t_r5+t_r10+v_r1+v_r5+v_r10)/6} ')

            del image_features  
            del text_features      



if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a resnet on opt_name')
    arg_parser.add_argument('--epochs', type=int, default=100)     # epoch 60
    arg_parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")  # 学习率0.00001
    arg_parser.add_argument('--bert_name', default='G:\Image-Text_Matching\MSSA\BERT\bert-base-uncased', type=str)

    args_re = arg_parser.parse_args()

    main(args_re)