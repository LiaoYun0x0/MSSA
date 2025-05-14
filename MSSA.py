from torch import nn
import torch.nn.functional as F
from resnet import resnet50 as resnet
import tokenizers   #将文本转换为模型可以处理的格格：将句子分割成单词或子词，并将他们转换为相应的索引。
from transformers import BertModel, BertConfig    #Transformer提供预训练的BERT模型
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
import torch
from ipdb import set_trace
from transformers import BertTokenizer
from transformer import LoFTREncoderLayer
import math
from typing import Tuple, Union
from collections import OrderedDict
import numpy as np
import copy
from PSCJA import Stage
'''MSSA '''

# -----------------------------------------------------------------------------------
def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


'''  1、特征提取  '''
class ImageEncoder(nn.Module):
    def __init__(self,arg):
        super(ImageEncoder, self).__init__()
        self.feature_extractor = resnet(pretrained=True,arg=arg)  #使用预训练的 ResNet 模型作为特征提取器
        #pretrained=True表示使用预训练的模型（权重）
        self.norm_layer = nn.BatchNorm2d(1)   #标准化特征图
        self.relu = nn.ReLU(inplace=True)

        #全连接层：将不同尺度的特征图转换为统一的特征向量
        self.fc2 = nn.Linear(512, 768)
        self.fc3 = nn.Linear(1024, 768)
        self.fc4 = nn.Linear(2048, 768)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        #特征提取
        '''
        feature_x8:[batch_size, 512, 28, 28]
        feature_x16:[batch_size, 1024, 14, 14]
        feature_x32:[batch_size, 2048, 7, 7]
        '''
        feature_x8, feature_x16, feature_x32= self.feature_extractor(x)

        # 将特征图展平为 [B, H*W, C] 形状
        B, C8, H8, W8 = feature_x8.shape
        B, C16, H16, W16 = feature_x16.shape
        B, C32, H32, W32 = feature_x32.shape
     
        
        # set_trace()     [B, H*W, C]  
        feature_x8_flat = feature_x8.view(B, H8 * W8, C8)  #[B, 28*28, 512]，即 [B, 784, 512]
        feature_x16_flat = feature_x16.view(B, H16 * W16, C16)  #[B, 14*14, 1024]，即 [B, 196, 1024]
        feature_x32_flat = feature_x32.view(B, H32 * W32, C32)  # [B, 7*7, 2048]，即 [B, 49, 2048]

        # 特征融合：对不同尺度的特征图进行池化，下采样到相同大小
        f2 = self.fc2(self.avgpool(feature_x8).view(B, C8))  # [batch_size, 512] -> [batch_size, 768]
        f3 = self.fc3(self.avgpool(feature_x16).view(B, C16))  # [batch_size, 1024] -> [batch_size, 768]
        f4 = self.fc4(self.avgpool(feature_x32).view(B, C32))  # [batch_size, 2048] -> [batch_size, 768]

        features =  f2 + f3 + f4     #特征融合    batch_size,768
     
        return features,feature_x8_flat, feature_x16_flat, feature_x32_flat


class bertmodel(BertModel):
    def __init__(self,config,add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)

    def forward(
            self,
            input_ids=None,   
            #image_feature=None,
            attention_mask=None,  
            key=None,
            token_type_ids=None,   
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:  # 64*40
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_ids.shape[0], input_ids.shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if key == False:
            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_key_values_length,
            )

      
            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            sequence_output = encoder_outputs[0]



            # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
            
          
        return sequence_output,embedding_output
        # return 0,embedding_output


#BERT提取文本特征
class Bert(nn.Module):
    '''
    sequence_output[batch_size, hidden_size]:    第一个token的特征向量(CLS token)
    sequence_output_embeddings[batch_size, hidden_size]:  所有token的特征向量的和         全局特征
    sequence_outputs[:, 1:, :] [batch_size, seq_length - 1, hidden_size]:  CLS token以外token的特征向量    局部特征
    '''
    def __init__(self, bert_name):
        super().__init__()
        # self.only_textembeddinmg = arg.only_textembeddinmg
        bert_config = BertConfig.from_pretrained(bert_name)  
        # set_trace()
        self.bert_model = bertmodel.from_pretrained(bert_name,config=bert_config)  
        bert_vocab = bert_name.join('/vocab.txt')
        #self.tokenizers = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')
        self.tokenizers = BertTokenizer.from_pretrained('G:\Image-Text_Matching\MSSA\BERT\bert-base-uncased')
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,text, attention_mask):
       
            
        sequence_outputs,sequence_outputs_all = self.bert_model(input_ids=text, attention_mask=attention_mask, key=False)
        # set_trace()
        sequence_output = sequence_outputs[:, 0, :]
       
        sequence_output_embeddings = torch.sum(sequence_outputs_all,dim=1)
        
        return sequence_output,sequence_output_embeddings,sequence_outputs[:, 1:, :]
       

# -----------------------------------------------------------------------------------
'''  2、模态内特征增强  '''

class TextAttention(nn.Module):
    '''
    input：[batch_size, seq_len-1, 768]
    output  out_easy: [batch_size, seq_len-1, 768] , out:[batch_size, seq_len-1, 768]
    '''
    def __init__(self):
        super(TextAttention, self).__init__()
        self.fc = nn.Linear(768, 768)  
        self.att = nn.Linear(768, 768)  
        self.sig = nn.Sigmoid()         
        self.fc2 = nn.Linear(768, 768) 

    def forward(self, input):

        out_easy = self.fc(input)  
        out_easy_att = self.sig(self.att(out_easy))  
        out_easy_ = out_easy * out_easy_att  
        out = self.fc2(out_easy_)  
        return out_easy, out

# -----------------------------------------------------------------------------------
class CrossAttention(nn.Module):
    #q,k,v:   [batch_size, seq_len, embed_dim]
    def __init__(self, embed_dim, num_heads,drop_prob=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.drop_path = DropPath(drop_prob) 
       
        self._init_weights()
        
    def _init_weights(self):
        scale = self.embed_dim ** -0.5
        proj_std = scale * ((2 * self.num_heads) ** -0.5)
        attn_std = scale
        nn.init.normal_(self.query_proj.weight, std=attn_std)
        nn.init.normal_(self.key_proj.weight, std=attn_std)
        nn.init.normal_(self.value_proj.weight, std=attn_std)
        nn.init.normal_(self.multihead_attn.in_proj_weight, std=attn_std)
        nn.init.normal_(self.multihead_attn.out_proj.weight, std=proj_std)
  
    def forward(self, query, key, value):
        
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        query = query.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        key = key.transpose(0, 1)      # [seq_len, batch_size, embed_dim]
        value = value.transpose(0, 1)  # [seq_len, batch_size, embed_dim]

        attn_output, attn_weights = self.multihead_attn(query, key, value)

        attn_output = attn_output.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        attn_output = self.drop_path(attn_output)
        attn_output = self.norm(attn_output + query.transpose(0,1))  
        return attn_output, attn_weights
    

class MatchingModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim, num_heads,drop_prob=0.1):
        super(MatchingModel, self).__init__()
        self.image_encoder = image_encoder
        self.bert = text_encoder
        self.avgpoolt0 = nn.AdaptiveAvgPool1d(32)
        self.avgpoolt1 = nn.AdaptiveAvgPool1d(16)
        self.avgpoolt2 = nn.AdaptiveAvgPool1d(8)
        self.avgpoolt3 = nn.AdaptiveAvgPool1d(1)
        self.avgpoolv0 = nn.AdaptiveAvgPool1d(16)
        self.down_img = nn.Conv1d(512+1024+2048, 768, 1)
        self.down_img1 = nn.Conv1d(768*2, 768, 1)
        self.down_img2 = nn.Conv1d(768, 256, 1)
        self.down_txt1 = nn.Conv1d(768, 256, 1)
    
        self.text_attention = TextAttention()
        self.cross_attention = CrossAttention(embed_dim, num_heads,drop_prob)

        self.image_proj_8 = nn.Linear(768, 512) #512
        self.image_proj_16 = nn.Linear(768, 1024) # 1024 
        self.image_proj_32 = nn.Linear(768, 2048) # 2048
        self.seeds_proj_8 = nn.Linear(768, 512) #512
        self.seeds_proj_16 = nn.Linear(768, 1024) # 1024 
        self.seeds_proj_32 = nn.Linear(768, 2048) # 2048

        self.text_proj = nn.Linear(embed_dim, 768)


        self.match_fc1 = nn.Linear(768, 1)
        self.match_fc2 = nn.Linear(768, 1)
        self.match_fc3 = nn.Linear(768, 1)
        self.cls_activation = nn.Sigmoid()
        self.drop_path = DropPath(drop_prob)
        encoder_layer = LoFTREncoderLayer(512, 8, 'linear')
        encoder_layer1 = LoFTREncoderLayer(1024, 8, 'linear')
        encoder_layer2 = LoFTREncoderLayer(2048, 8, 'linear')
        encoder_layer3 = LoFTREncoderLayer(768, 8, 'linear')

        encoder_layert1 = Stage(512,512,2, 8)
        encoder_layert2 = Stage(1024,1024, 2,8)
        encoder_layert3 =  Stage(2048,2048, 2,8)
        encoder_layert4 =  Stage(768,768,2, 8)
        self.layer_names = ['self','cross'] * 2
        self.layer_names2 = ['self','cross'] * 2
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names)//2)])
        self.layerst = nn.ModuleList([copy.deepcopy(encoder_layert1) for _ in range(len(self.layer_names)//2)])
        self.layers = self.layerst.extend(self.layers)
        self.layers1 = nn.ModuleList([self.layers[0], self.layers[2],
                                      self.layers[1],self.layers[3]])
                                    #   self.layers[2],self.layers[5]])
                                    #   self.layers[3],self.layers[7]])
                                    #   self.layers[4],self.layers[9]])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer1) for _ in range(len(self.layer_names2)//2)])
        self.layerst = nn.ModuleList([copy.deepcopy(encoder_layert2) for _ in range(len(self.layer_names2)//2)])
        self.layers = self.layerst.extend(self.layers)
        self.layers2 = nn.ModuleList([self.layers[0], self.layers[2],
                                      self.layers[1],self.layers[3]])
                                      # self.layers[2],self.layers[5]])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer2) for _ in range(len(self.layer_names2)//2)])
        self.layerst = nn.ModuleList([copy.deepcopy(encoder_layert3) for _ in range(len(self.layer_names2)//2)])
        self.layers = self.layerst.extend(self.layers)
        self.layers3 = nn.ModuleList([self.layers[0], self.layers[2],
                                      self.layers[1],self.layers[3]])
                                      # self.layers[2],self.layers[5]])
        self.layer_names1 = ['self','cross'] * 5
        self.layers4 = nn.ModuleList([copy.deepcopy(encoder_layer3) for _ in range(len(self.layer_names1))])
        
        self.seed_tokens = nn.Parameter(torch.randn(20, 768))
        self.register_parameter('seed_tokens', self.seed_tokens)
        self.topic_drop = nn.Dropout1d(p=0.1)
        # self.topic_drop = nn.Dropout1d(p=0.1)
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, text, attention_mask, image,num_seed=24):
        # 特征提取
        image_features, feature_x8, feature_x16, feature_x32 = self.image_encoder(image)
        # print(image_features.shape)
        '''
        feature_x8:  B, 784, 512
        feature_x16: B,196,1024
        feature_x32: B, 49, 2048
        '''
      
        seeds = self.seed_tokens.unsqueeze(0).repeat(text.size(0), 2, 1)
        # print(seeds.shape)
        seeds = self.topic_drop(seeds)
        seeds8 = self.seeds_proj_8(seeds) # 512
        seeds16 = self.seeds_proj_16(seeds) # 1024
        seeds32 = self.seeds_proj_32(seeds) # 2048
        seeds08 =seeds8[:,:20,:]
        seeds18 =seeds8[:,20:,:]
        seeds016 =seeds16[:,:20,:]
        seeds116 =seeds16[:,20:,:]
        seeds032 =seeds32[:,:20,:]
        seeds132 =seeds32[:,20:,:]
        sequence_output, sequence_output_embeddings, local_feature = self.bert(text, attention_mask)
        text32 = torch.concat((sequence_output.unsqueeze(1),sequence_output_embeddings.unsqueeze(1),local_feature),1)
        text16 = self.avgpoolt1(text32.permute(0,2,1)).permute(0,2,1)
        text8 = self.avgpoolt2(text16.permute(0,2,1)).permute(0,2,1)
        text8 = self.image_proj_8(text8) # 512
        text16 = self.image_proj_16(text16) # 1024
        text32 = self.image_proj_32(text32) # 2048

        for  layer, name in zip(self.layers1, self.layer_names):
            if name == 'self':
                feature_x8 = layer(feature_x8,28,28)
                feature_x8 = torch.nn.functional.normalize(feature_x8, dim=-1)
            elif name == 'cross':
                text8 = layer(text8,feature_x8)
                text8 = torch.nn.functional.normalize(text8, dim=-1)
                seeds08 = layer(seeds08, feature_x8)
                seeds18 = layer(seeds18, text8)
                seeds08 = layer(seeds08,seeds18, None, None)
                seeds18 = layer(seeds18,seeds08, None, None) 
                seeds08 = torch.nn.functional.normalize(seeds08, dim=-1)
                seeds18 = torch.nn.functional.normalize(seeds08, dim=-1)
      
        for layer, name in zip(self.layers2, self.layer_names2):
            if name == 'self':
                feature_x16 = layer(feature_x16,14,14)
                feature_x16 = torch.nn.functional.normalize(feature_x16, dim=-1)
            elif name == 'cross':
                text16 = layer(text16,feature_x16)
                text16 = torch.nn.functional.normalize(text16, dim=-1)
                seeds016 = layer(seeds016, feature_x16)
                seeds116 = layer(seeds116, text16)
                seeds016 = layer(seeds016,seeds116, None, None)
                seeds116 = layer(seeds116,seeds016, None, None) 
                seeds016 = torch.nn.functional.normalize(seeds016, dim=-1)
                seeds116 = torch.nn.functional.normalize(seeds116, dim=-1)

        for layer, name in zip(self.layers3, self.layer_names2):
            if name == 'self':
                    feature_x32 = layer(feature_x32,7,7)
                    feature_x32 = torch.nn.functional.normalize(feature_x32, dim=-1)
            elif name == 'cross':
                text32 = layer(text32,feature_x32)
                text32 = torch.nn.functional.normalize(text32, dim=-1) 
                seeds032 = layer(seeds032, feature_x32)
                seeds132 = layer(seeds132, text32)
                seeds032 = layer(seeds032,seeds132, None, None)
                seeds132 = layer(seeds132,seeds032, None, None) 
                seeds032 = torch.nn.functional.normalize(seeds032, dim=-1)
                seeds132 = torch.nn.functional.normalize(seeds132, dim=-1) 
        v = torch.concat((self.avgpoolv0(feature_x32.permute(0,2,1)).permute(0,2,1),self.avgpoolv0(feature_x8.permute(0,2,1)).permute(0,2,1),self.avgpoolv0(feature_x16.permute(0,2,1)).permute(0,2,1)),-1)
        t = torch.concat((self.avgpoolv0(text32.permute(0,2,1)).permute(0,2,1),self.avgpoolv0(text8.permute(0,2,1)).permute(0,2,1),self.avgpoolv0(text16.permute(0,2,1)).permute(0,2,1)),-1)
        v= self.down_img(v.permute(0,2,1)).permute(0,2,1)
        t= self.down_img(t.permute(0,2,1)).permute(0,2,1)
        v = self.avgpoolt3(v.permute(0,2,1)).permute(0,2,1)
        v= torch.concat((image_features.unsqueeze(1),v),1)
        # t = torch.concat((text8,text16,text32),1)
        t = self.avgpoolt3(t.permute(0,2,1)).permute(0,2,1)
        t= torch.concat((sequence_output.unsqueeze(1),t),1)
        seeds0 = torch.concat((seeds032,seeds016,seeds08),-1)
        seeds1 = torch.concat((seeds132,seeds116,seeds18),-1)
        seeds0= self.down_img(seeds0.permute(0,2,1)).permute(0,2,1) 
        seeds1= self.down_img(seeds1.permute(0,2,1)).permute(0,2,1) 
        seeds0 = self.avgpoolt3(seeds0.permute(0,2,1)).permute(0,2,1)
        seeds1 = self.avgpoolt3(seeds1.permute(0,2,1)).permute(0,2,1)
        seeds0= torch.concat((image_features.unsqueeze(1),seeds0),-1)
        seeds1= torch.concat((sequence_output.unsqueeze(1),seeds1),-1)
        seeds0 = self.down_img1(seeds0.permute(0,2,1)).permute(0,2,1)
        seeds1 = self.down_img1(seeds1.permute(0,2,1)).permute(0,2,1)
        # seeds0= torch.concat((image_features.unsqueeze(1),seeds0),-1)
        # seeds1= torch.concat((sequence_output.unsqueeze(1),seeds1),-1)
        for id,layer, name in zip(range(len(self.layer_names)),self.layers4, self.layer_names1):
            if name == 'self':
                # print(feature_x8.shape) 
                    v = layer(v,v)
                    v = torch.nn.functional.normalize(v, dim=-1)
                    t = layer(t, t)
                    t = torch.nn.functional.normalize(t, dim=-1)   
            elif name == 'cross': 
                if id%2==0:
                    seeds0 = layer(seeds0,v, None, None)
                    seeds0 = layer(seeds0,t, None, None) 
                    seeds0 = layer(seeds0,seeds1, None, None)
                else:
                    seeds1 = layer(seeds1,t, None, None) 
                    seeds1 = layer(seeds1,v, None, None) 
                    seeds1 = layer(seeds1,seeds0, None, None) 
        seeds0 = torch.nn.functional.normalize(seeds0, dim=-1)  
        seeds1 = torch.nn.functional.normalize(seeds1, dim=-1)
        seeds0 = self.down_img2(seeds0.permute(0,2,1)).permute(0,2,1)
        seeds1 = self.down_txt1(seeds1.permute(0,2,1)).permute(0,2,1)  
        seeds0= torch.concat((image_features.unsqueeze(1),seeds0),-1)
        seeds1= torch.concat((sequence_output.unsqueeze(1),seeds1),-1)
    
        return seeds0, seeds1

    # def process_stage(self, feature_x8, feature_x16, feature_x32, local_feature, stage,
    #                   prev_image_text_cross_attn_output=None, prev_text_image_cross_attn_output=None):
        
        
    #     # 模态内特征增强
    #     image_attn_output = self.image_attention(selected_feature)
    #     text_attn_output, _ = self.text_attention(local_feature)
        
      
    #     return image_attn_output, text_attn_output
 


