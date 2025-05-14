import copy
import torch
import torch.nn as nn
from linear_attention import LinearAttention, FullAttention
import math
from einops import rearrange
import torch.nn.functional as F


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )
        # self.relu = nn.Sigmoid()
        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(LocalFeatureTransformer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.layer_names = config['layer_names']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


class TopicFormer(nn.Module):

    def __init__(self, config):
        super(TopicFormer, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.global_avg_pool5 = nn.AdaptiveAvgPool2d((5, 5))
        self.global_avg_pool10 = nn.AdaptiveAvgPool2d((10, 10))
        self.global_avg_pool20 = nn.AdaptiveAvgPool2d((20, 20))
        self.avgpool = nn.AdaptiveAvgPool1d(256)
        self.avgpool1 = nn.AdaptiveAvgPool1d(128)
        self.avgpool2 = nn.AdaptiveAvgPool1d(192)
        # self.avgpool3 = nn.AvgPool1d(3)
        # self.avgpool4 = nn.AvgPool1d(6)
        self.layer_names = config['layer_names_t']
        self.layer_names1 = config['layer_names_t1']
        self.layer_names2 = config['layer_names_t2']
        self.layer_names3 = config['layer_names_t3']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        encoder_layer1 = LoFTREncoderLayer(config['d_model'] // 2, config['nhead'], config['attention'])
        encoder_layer2 = LoFTREncoderLayer(config['d_model'] // 4 * 3, config['nhead'], config['attention'])
        encoder_layer3 = LoFTREncoderLayer(config['d_model'] // 4 * 3, config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self.layers1 = nn.ModuleList([copy.deepcopy(encoder_layer1) for _ in range(len(self.layer_names1))])
        self.layers2 = nn.ModuleList([copy.deepcopy(encoder_layer2) for _ in range(len(self.layer_names2))])
        self.layers3 = nn.ModuleList([copy.deepcopy(encoder_layer3) for _ in range(len(self.layer_names3))])

        # if config['n_samples'] > 0:
        self.feat_aug = nn.ModuleList(
            [copy.deepcopy(encoder_layer2) for _ in range(2 * config['n_topic_transformers'])])
        self.n_iter_topic_transformer = config['n_topic_transformers']

        self.seed_tokens = nn.Parameter(torch.randn(config['n_topics'], config['d_model']))
        self.register_parameter('seed_tokens', self.seed_tokens)
        self.topic_drop = nn.Dropout1d(p=0.1)
        self.n_samples = config['n_samples']
        # self.avgpool = nn.AdaptiveAvgPool1d(160)
        # self.avgpool_1 = nn.AdaptiveAvgPool1d(96)
        self.norm_feat = nn.LayerNorm(576)
        # self.fea_down_t0 = conv1x1(config['d_model'] + config['d_model_fusion'] * 2, config['d_model_fusion'] * 2)
        # self.fea_down_t1 = conv1x1(config['d_model_fusion'] * 2, config['d_model_fusion'])
        # self.fea_down_t2 = conv1x1(config['d_model_fusion'], config['d_model'])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sample_topic(self, prob_topics, topics, L):
        prob_topics0, prob_topics1 = prob_topics[:, :L], prob_topics[:, L:]
        topics0, topics1 = topics[:, :L], topics[:, L:]

        theta0 = F.normalize(prob_topics0.sum(dim=1), p=1, dim=-1)  # [N, K]
        theta1 = F.normalize(prob_topics1.sum(dim=1), p=1, dim=-1)
        theta = F.normalize(theta0 * theta1, p=1, dim=-1)
        if self.n_samples == 0:
            return None
        if self.training:
            sampled_inds = torch.multinomial(theta, self.n_samples)
            sampled_values = torch.gather(theta, dim=-1, index=sampled_inds)
        else:
            sampled_values, sampled_inds = torch.topk(theta, self.n_samples, dim=-1)
        sampled_topics0 = torch.gather(topics0, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics0.shape[1], 1))
        sampled_topics1 = torch.gather(topics1, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics1.shape[1], 1))
        return sampled_topics0, sampled_topics1

    def reduce_feat(self, feat, topick, N, C):
        len_topic = topick.sum(dim=-1).int()
        max_len = len_topic.max().item()
        selected_ids = topick.bool()
        resized_feat = torch.zeros((N, max_len, C), dtype=torch.float, device=feat.device)
        new_mask = torch.zeros_like(resized_feat[..., 0]).bool()
        for i in range(N):
            new_mask[i, :len_topic[i]] = True
        resized_feat[new_mask, :] = feat[selected_ids, :]
        return resized_feat, new_mask, selected_ids

    def forward(self, feats16_0, feats16_1, feats8_0, feats8_1, feats4_0, feats4_1, t64, t32, t16, data):
        mask = mask0 = mask1 = None
        N16, L16, C16 = feats16_0.shape
        N8, L8, C8 = feats8_0.shape
        N4, L4, C4 = feats4_0.shape
        NT64, LT64, CT64 = t64.shape
        NT32, LT32, CT32 = t32.shape
        NT16, LT16, CT16 = t16.shape
        H16 = int(math.sqrt(L16))
        H8 = int(math.sqrt(L8))
        H4 = int(math.sqrt(L4))
        HT64 = int(math.sqrt(LT64))
        HT32 = int(math.sqrt(LT32))
        HT16 = int(math.sqrt(LT16))

        kernel_size0 = H16 // HT32
        kernel_size1 = H4 // HT16
        kernel_size2 = H8 // HT64

        L = L8

        conf_matrix_16 = torch.einsum("nlc,nsc->nls", feats16_0, feats16_1) / C16 ** .5
        conf_matrix_16_f = conf_matrix_16.unsqueeze(1)
        # print(conf_matrix_16_f.shape)
        conf_matrix_16_f = F.interpolate(conf_matrix_16_f, scale_factor=4., mode='bilinear',
                                         align_corners=True)
        data.update({"conf_matrix_16_f": conf_matrix_16_f})
        # print(conf_matrix_16_f.shape)
        conf_matrix_16 = F.softmax(conf_matrix_16, 1) * F.softmax(conf_matrix_16, 2)
        conf_matrix_16_idx = torch.argmax(conf_matrix_16, -1)
        # print(conf_matrix_16_idx.shape)
        conf_matrix_16_idy = torch.argmax(conf_matrix_16, -2)
        feats16_1_ = feats16_1[:, conf_matrix_16_idx[-1]]
        feats16_t = torch.reshape((feats16_0 + feats16_1_) / 2,
                                  (feats16_0.shape[0], 20, 20, feats16_0.shape[2])).permute(
            0, 3, 1, 2)  # 2 80 80 256*2
        feats16_t = self.global_avg_pool10(feats16_t).permute(0, 2, 3, 1)  # 1,256,10,10
        feats16_t = torch.reshape(feats16_t, (NT32, LT32, feats16_t.shape[-1]))
        t32 = torch.concat((t32, feats16_t), -1)
        # print(t32.shape)
        feats16_0 = self.avgpool(torch.concat((feats16_0, conf_matrix_16_idx.unsqueeze(-1)), -1))
        feats16_1 = self.avgpool(torch.concat((feats16_1, conf_matrix_16_idy.unsqueeze(-1)), -1))

        # t32 = torch.reshape(t32, (t32.shape[0], 10, 10, t32.shape[2])).permute(0, 3, 1, 2)
        # t32 = self.global_avg_pool5(t32)  # 1,256,20,20
        t32 = self.avgpool(t32)
        t32 = t32.unsqueeze(2).squeeze(0)
        # print(t32.shape)
        # if mask0 is not None:
        #     mask = torch.cat((mask0, mask1), dim=-1)
        # else:
        #     mask = None

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'seed':
                # t32s = []
                feats16_1 = feats16_1[:, conf_matrix_16_idx[-1]]
                feats16 = torch.concat((feats16_0, feats16_1), 0)
                feats16 = torch.reshape(feats16, (feats16.shape[0], 20, 20, feats16.shape[2])).permute(0, 3, 1, 2)
                feats16_u = F.unfold(feats16, kernel_size=(kernel_size0, kernel_size0), stride=kernel_size0,
                                     padding=0)
                feats16_u = rearrange(feats16_u, 'n (c ww) l -> n l ww c', ww=kernel_size0 ** 2)
                feats16_u = torch.concat((feats16_u[0], feats16_u[1]), -1)
                # print(feats16_u.shape)
                t32 = layer(t32.squeeze(0), self.avgpool(feats16_u))
                # print(t32.shape)
                # for ti, feati in zip(t32.squeeze(0), self.avgpool(feats16_u)):
                #     # feats16_u=self.avgpool(feats16_u)
                #     #     feat = torch.cat((feat0, feat1), dim=1)
                #     # print(ti.shape)
                #     # print(feati.shape)
                #     ti = layer(ti.unsqueeze(0), feati.unsqueeze(0), None, None)
                #     t32s.append(ti)
                # t32 = torch.stack(t32s).squeeze(1)
                # print(t32.shape)
            elif name == 'feat':
                feats16_0 = layer(feats16_0, t32.permute(1, 0, 2), None, None)
                feats16_1 = layer(feats16_1, t32.permute(1, 0, 2), None, None)
        feats16 = torch.concat((feats16_0, feats16_1), 0)
        feats16 = torch.reshape(feats16, (feats16.shape[0], 20, 20, feats16.shape[2])).permute(0, 3, 1, 2)
        t32 = torch.reshape(t32, (NT32, HT32, HT32, t32.shape[-1])).permute(0, 3, 1, 2)
        # print(t32.shape)
        # print(feats16.shape)

        conf_matrix_4 = torch.einsum("nlc,nsc->nls", feats4_0, feats4_1) / C4 ** .5
        conf_matrix_4_f = conf_matrix_4.unsqueeze(1)
        conf_matrix_4_f = F.interpolate(conf_matrix_4_f, scale_factor=0.25, mode='bilinear',
                                        align_corners=True)
        data.update({"conf_matrix_4_f": conf_matrix_4_f})
        conf_matrix_4 = F.softmax(conf_matrix_4, 1) * F.softmax(conf_matrix_4, 2)
        conf_matrix_4_idx = torch.argmax(conf_matrix_4, -1)
        conf_matrix_4_idy = torch.argmax(conf_matrix_4, -2)
        feats4_1_ = feats4_1[:, conf_matrix_4_idx[-1]]
        feats4_t = torch.reshape((feats4_0 + feats4_1_) / 2, (feats4_0.shape[0], 80, 80, feats4_0.shape[2])).permute(
            0, 3, 1, 2)  # 2 80 80 256*2
        feats4_t = self.global_avg_pool20(feats4_t).permute(0, 2, 3, 1)  # 1,256,5,5
        feats4_t = torch.reshape(feats4_t, (NT16, LT16, feats4_t.shape[-1]))
        t16 = torch.concat((t16, feats4_t), -1)
        # feats4_0_ = feats4_0[:, conf_matrix_4_idy[-1]]
        # feats4_x = torch.concat((feats4_0_, feats4_1_), -1)
        # feats4_y = torch.concat((feats4_1_, feats4_0_), -1)
        feats4_0 = torch.concat((feats4_0, conf_matrix_4_idx.unsqueeze(-1)), -1)
        feats4_1 = torch.concat((feats4_1, conf_matrix_4_idy.unsqueeze(-1)), -1)
        t16 = self.avgpool1(t16)
        t16 = t16.unsqueeze(2).squeeze(0)

        for layer, name in zip(self.layers1, self.layer_names1):
            if name == 'seed':
                # t16s = []
                feats4_1 = feats4_1[:, conf_matrix_4_idx[-1]]
                feats4 = torch.concat((feats4_0, feats4_1), 0)
                feats4 = torch.reshape(feats4, (feats4.shape[0], 80, 80, feats4.shape[2])).permute(0, 3, 1, 2)
                feats4_u = F.unfold(feats4, kernel_size=(kernel_size1, kernel_size1), stride=kernel_size1,
                                    padding=0)
                feats4_u = rearrange(feats4_u, 'n (c ww) l -> n l ww c', ww=kernel_size1 ** 2)
                feats4_u = torch.concat((feats4_u[0], feats4_u[1]), -1)
                t16 = layer(t16, self.avgpool1(feats4_u), None, None)
                # print(feats16_u.shape)
                # for ti, feati in zip(t16, self.avgpool1(feats4_u)):
                #     # feats16_u=self.avgpool(feats16_u)
                #     #     feat = torch.cat((feat0, feat1), dim=1)
                #     # print(ti.shape)
                #     # print(feati.shape)
                #     ti = layer(ti.unsqueeze(0), feati.unsqueeze(0), None, None)
                #     t16s.append(ti)
                # t16 = torch.stack(t16s).squeeze(1)

            elif name == 'feat':
                feats4_0 = layer(self.avgpool1(feats4_0), t16.permute(1, 0, 2), None, None)
                feats4_1 = layer(self.avgpool1(feats4_1), t16.permute(1, 0, 2), None, None)
        feats4 = torch.concat((feats4_0, feats4_1), 0)
        feats4 = torch.reshape(feats4, (feats4.shape[0], 80, 80, feats4.shape[2])).permute(0, 3, 1, 2)
        t16 = torch.reshape(t16, (NT16, HT16, HT16, t16.shape[-1])).permute(0, 3, 1, 2)
        # print(t16.shape)
        # print(feats4.shape)

        conf_matrix_8 = torch.einsum("nlc,nsc->nls", feats8_0, feats8_1) / C8 ** .5
        conf_matrix_8_f = conf_matrix_8.unsqueeze(1)

        data.update({"conf_matrix_8_f": conf_matrix_8_f})
        conf_matrix_8 = F.softmax(conf_matrix_8, 1) * F.softmax(conf_matrix_8, 2)
        conf_matrix_8_idx = torch.argmax(conf_matrix_8, -1)
        conf_matrix_8_idy = torch.argmax(conf_matrix_8, -2)
        feats8_1_ = feats8_1[:, conf_matrix_8_idx[-1]]
        feats8_t = torch.reshape((feats8_0 + feats8_1_) / 2, (feats8_0.shape[0], 40, 40, feats8_0.shape[2])).permute(
            0, 3, 1, 2)  # 2 80 80 256*2
        feats8_t = self.global_avg_pool5(feats8_t).permute(0, 2, 3, 1)  # 1,256,5,5
        feats8_t = torch.reshape(feats8_t, (NT64, LT64, feats8_t.shape[-1]))
        t64 = torch.concat((t64, feats8_t), -1)
        # feats4_0_ = feats4_0[:, conf_matrix_4_idy[-1]]
        # feats4_x = torch.concat((feats4_0_, feats4_1_), -1)
        # feats4_y = torch.concat((feats4_1_, feats4_0_), -1)
        feats8_0 = torch.concat((feats8_0, conf_matrix_8_idx.unsqueeze(-1)), -1)
        feats8_1 = torch.concat((feats8_1, conf_matrix_8_idy.unsqueeze(-1)), -1)
        t64 = self.avgpool2(t64)
        t64 = t64.unsqueeze(2).squeeze(0)
        # print(feats8_0.shape)
        # print(t64.shape)
        for layer, name in zip(self.layers2, self.layer_names2):
            if name == 'seed':
                # t64s = []
                feats8_1 = feats8_1[:, conf_matrix_8_idx[-1]]
                feats8 = torch.concat((feats8_0, feats8_1), 0)
                feats8 = torch.reshape(feats8, (feats8.shape[0], 40, 40, feats8.shape[2])).permute(0, 3, 1, 2)
                feats8_u = F.unfold(feats8, kernel_size=(kernel_size2, kernel_size2), stride=kernel_size2,
                                    padding=0)
                feats8_u = rearrange(feats8_u, 'n (c ww) l -> n l ww c', ww=kernel_size2 ** 2)
                feats8_u = torch.concat((feats8_u[0], feats8_u[1]), -1)
                # print(feats16_u.shape)
                t64 = layer(t64, self.avgpool2(feats8_u))
                # for ti, feati in zip(t64, self.avgpool2(feats8_u)):
                #     # feats16_u=self.avgpool(feats16_u)
                #     #     feat = torch.cat((feat0, feat1), dim=1)
                #     # print(ti.shape)
                #     # print(feati.shape)
                #     ti = layer(ti.unsqueeze(0), feati.unsqueeze(0), None, None)
                #     t64s.append(ti)
                # t64 = torch.stack(t64s).squeeze(1)
                # print(t32.shape)
            elif name == 'feat':
                feats8_0 = layer(self.avgpool2(feats8_0), t64.permute(1, 0, 2), None, None)
                feats8_1 = layer(self.avgpool2(feats8_1), t64.permute(1, 0, 2), None, None)
        feats8 = torch.concat((feats8_0, feats8_1), 0)
        feats8 = torch.reshape(feats8, (feats8.shape[0], 40, 40, feats8.shape[2])).permute(0, 3, 1, 2)
        t64 = torch.reshape(t64, (NT64, HT64, HT64, t64.shape[-1])).permute(0, 3, 1, 2)
        # print(t64.shape)
        # print(t32.shape)
        # print(t16.shape)
        # print(feats16.shape)
        # print(feats8.shape)
        # print(feats4.shape)
        # torch.Size([1, 5, 5, 192])
        # torch.Size([1, 10, 10, 256])
        # torch.Size([1, 20, 20, 128])
        # torch.Size([2, 256, 20, 20])
        # torch.Size([2, 192, 40, 40])
        # torch.Size([2, 128, 80, 80])
        t16 = self.global_avg_pool5(t16)  # 1,256,5,5
        t32 = self.global_avg_pool5(t32)  # 1,256,5,5
        feats16 = F.interpolate(feats16, scale_factor=2, mode='bilinear', align_corners=True)
        feats4 = F.interpolate(feats4, scale_factor=0.5, mode='bilinear', align_corners=True)
        t64 = torch.reshape(torch.concat((t64, t32, t16), 1).permute(0, 2, 3, 1), (NT64, LT64, -1)).permute(1, 0, 2) / CT64 ** 0.5
        # t64 = self.norm_feat(t64)
        feats8 = torch.reshape(torch.concat((feats8, feats16, feats4), 1).permute(0, 2, 3, 1),
                               (N8 * 2, L8, -1))
        feats8_0 = feats8[0].unsqueeze(0)
        feats8_1 = feats8[1].unsqueeze(0)
        # print(t64.shape)
        # print(feats8.shape)
        # print(feats8_0.shape)
        # print(feats8_1.shape)

        for layer, name in zip(self.layers3, self.layer_names3):
            if name == 'seed':
                # t64s = []
                feats8_1 = feats8_1[:, conf_matrix_8_idx[-1]]
                feats8 = torch.concat((feats8_0, feats8_1), 0)
                feats8 = torch.reshape(feats8, (feats8.shape[0], 40, 40, feats8.shape[2])).permute(0, 3, 1, 2)
                feats8_u = F.unfold(feats8, kernel_size=(kernel_size2, kernel_size2), stride=kernel_size2,
                                    padding=0)
                feats8_u = rearrange(feats8_u, 'n (c ww) l -> n l ww c', ww=kernel_size2 ** 2)
                feats8_u = torch.concat((feats8_u[0], feats8_u[1]), -1)
                # print(feats16_u.shape)
                t64 = layer(self.avgpool2(t64), self.avgpool2(feats8_u))
                # for ti, feati in zip(self.avgpool2(t64), self.avgpool2(feats8_u)):
                #     # feats16_u=self.avgpool(feats16_u)
                #     #     feat = torch.cat((feat0, feat1), dim=1)
                #     # print(ti.shape)
                #     # print(feati.shape)
                #     # print(ti.unsqueeze(0))
                #     # print(feati.unsqueeze(0))
                #     ti = layer(ti.unsqueeze(0), feati.unsqueeze(0), None, None)
                #     t64s.append(ti)
                # t64 = torch.stack(t64s).squeeze(1)
                # print(t32.shape)
            elif name == 'feat':
                feats8_0 = layer(self.avgpool2(feats8_0), t64.permute(1, 0, 2), None, None)
                feats8_1 = layer(self.avgpool2(feats8_1), t64.permute(1, 0, 2), None, None)
        feats8 = torch.concat((feats8_0, feats8_1), 1)
        t64 = t64.permute(1, 0, 2)
        # feats8 = torch.reshape(feats8, (feats8.shape[0], 40, 40, feats8.shape[2])).permute(0, 3, 1, 2)
        # feats8 = torch.reshape(feats8, (feats8.shape[0], 40, 40, feats8.shape[2])).permute(0, 3, 1, 2)
        # t64 = torch.reshape(t64, (NT64, HT64, HT64, t64.shape[-1])).permute(0, 3, 1, 2)
        # t64 = torch.reshape(t64, (NT64, HT64, HT64, t64.shape[-1])).permute(0, 3, 1, 2)
        # print(feats8.shape)
        # print(t64.shape)
        seeds = t64
        feat = feats8
        dmatrix = torch.einsum("nmd,nkd->nmk", feat, seeds) / C8 ** .5
        prob_topics = F.softmax(dmatrix, dim=-1)

        feat_topics = torch.zeros_like(dmatrix).scatter_(-1, torch.argmax(dmatrix, dim=-1, keepdim=True), 1.0)

        if mask is not None:
            feat_topics = feat_topics * mask.unsqueeze(-1)
            prob_topics = prob_topics * mask.unsqueeze(-1)

        sampled_topics = self.sample_topic(prob_topics.detach(), feat_topics, L)
        if sampled_topics is not None:
            updated_feat0, updated_feat1 = torch.zeros_like(feats8_0), torch.zeros_like(feats8_1)
            s_topics0, s_topics1 = sampled_topics
            for k in range(s_topics0.shape[-1]):
                topick0, topick1 = s_topics0[..., k], s_topics1[..., k]  # [N, L+S]
                if (topick0.sum() > 0) and (topick1.sum() > 0):
                    new_feat0, new_mask0, selected_ids0 = self.reduce_feat(feats8_0, topick0, N8, C8 // 4 * 3)
                    new_feat1, new_mask1, selected_ids1 = self.reduce_feat(feats8_1, topick1, N8, C8 // 4 * 3)
                    for idt in range(self.n_iter_topic_transformer):
                        new_feat0 = self.feat_aug[idt * 2](new_feat0, new_feat0, new_mask0, new_mask0)
                        new_feat1 = self.feat_aug[idt * 2](new_feat1, new_feat1, new_mask1, new_mask1)
                        new_feat0 = self.feat_aug[idt * 2 + 1](new_feat0, new_feat1, new_mask0, new_mask1)
                        new_feat1 = self.feat_aug[idt * 2 + 1](new_feat1, new_feat0, new_mask1, new_mask0)
                    updated_feat0[selected_ids0, :] = new_feat0[new_mask0, :]
                    updated_feat1[selected_ids1, :] = new_feat1[new_mask1, :]

            feats8_0 = (1 - s_topics0.sum(dim=-1, keepdim=True)) * feats8_0 + updated_feat0
            feats8_1 = (1 - s_topics1.sum(dim=-1, keepdim=True)) * feats8_1 + updated_feat1
        else:
            for idt in range(self.n_iter_topic_transformer * 2):
                feats8_0 = self.feat_aug[idt](feats8_0, seeds, mask0, None)
                feats8_1 = self.feat_aug[idt](feats8_1, seeds, mask1, None)

        if self.training:
            # topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:])
            # topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:]) / C ** .5
            # print(topic_matrix)
            topic_matrix_img0 = feat_topics[:, :L]
            Nt, Ct, Ht, Wt = topic_matrix_img0.shape[0], topic_matrix_img0.shape[2], int(math.sqrt(
                topic_matrix_img0.shape[1])), int(math.sqrt(topic_matrix_img0.shape[1]))

            topic_matrix_img0 = torch.reshape(topic_matrix_img0, (Nt, Ct, Ht, Wt))

            topic_matrix_img0_unfold = F.unfold(topic_matrix_img0, kernel_size=(3, 3), stride=1, padding=1)
            topic_matrix_img0_unfold = rearrange(topic_matrix_img0_unfold, 'n (c ww) l -> n l ww c',
                                                 ww=3 ** 2)  # 2*144*9*256
            topic_matrix_img0_unfold = torch.mean(torch.transpose(topic_matrix_img0_unfold, -1, -2), -1)
            topic_matrix_img0_unfold_mask = topic_matrix_img0_unfold > 0.5
            # topic_matrix_img0_unfold_mask = topic_matrix_img0_unfold > 0
            topic_matrix_img0_unfold_mask = topic_matrix_img0_unfold_mask.byte()
            topic_matrix_img0_unfold_mask = torch.reshape(topic_matrix_img0_unfold_mask, (Nt, Ht * Wt, Ct))
            topic_matrix_img0 = torch.reshape(topic_matrix_img0, (Nt, Ht * Wt, Ct))

            topic_matrix_img0 = topic_matrix_img0 * topic_matrix_img0_unfold_mask

            topic_matrix_img1 = feat_topics[:, L:]

            topic_matrix_img1 = torch.reshape(topic_matrix_img1, (Nt, Ct, Ht, Wt))

            topic_matrix_img1_unfold = F.unfold(topic_matrix_img1, kernel_size=(3, 3), stride=1, padding=1)
            topic_matrix_img1_unfold = rearrange(topic_matrix_img1_unfold, 'n (c ww) l -> n l ww c',
                                                 ww=3 ** 2)  # 2*144*9*256
            topic_matrix_img1_unfold = torch.mean(torch.transpose(topic_matrix_img1_unfold, -1, -2), -1)
            topic_matrix_img1_unfold_mask = topic_matrix_img1_unfold > 0.5
            # topic_matrix_img1_unfold_mask = topic_matrix_img1_unfold > 0
            topic_matrix_img1_unfold_mask = topic_matrix_img1_unfold_mask.byte()
            topic_matrix_img1 = torch.reshape(topic_matrix_img1, (Nt, Ht * Wt, Ct))
            topic_matrix_img1_unfold_mask = torch.reshape(topic_matrix_img1_unfold_mask, (Nt, Ht * Wt, Ct))
            topic_matrix_img1 = topic_matrix_img1 * topic_matrix_img1_unfold_mask
            topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:])
            # print(topic_matrix)
            topic_matrix_match = {"img0": topic_matrix_img0, "img1": topic_matrix_img1}

        else:
            topic_matrix = {"img0": feat_topics[:, :L], "img1": feat_topics[:, L:]}
            topic_matrix_match = {"img0": feat_topics[:, :L], "img1": feat_topics[:, L:]}

        return feats8_0, feats8_1, topic_matrix, topic_matrix_match, seeds


class TopicFormer1(nn.Module):

    def __init__(self, config):
        super(TopicFormer1, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.global_avg_pool5 = nn.AdaptiveAvgPool2d((5, 5))
        self.global_avg_pool10 = nn.AdaptiveAvgPool2d((10, 10))
        self.global_avg_pool20 = nn.AdaptiveAvgPool2d((20, 20))
        self.avgpool = nn.AdaptiveAvgPool1d(256)
        self.avgpool1 = nn.AdaptiveAvgPool1d(128)
        self.avgpool2 = nn.AdaptiveAvgPool1d(192)
        # self.avgpool3 = nn.AvgPool1d(3)
        # self.avgpool4 = nn.AvgPool1d(6)
        self.layer_names = config['layer_names_t']
        self.layer_names1 = config['layer_names_t1']
        self.layer_names2 = config['layer_names_t2']
        self.layer_names3 = config['layer_names_t3']
        encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        encoder_layer1 = LoFTREncoderLayer(config['d_model'] // 2, config['nhead'], config['attention'])
        encoder_layer2 = LoFTREncoderLayer(config['d_model'] // 4 * 3, config['nhead'], config['attention'])
        encoder_layer3 = LoFTREncoderLayer(config['d_model'] // 4 * 3, config['nhead'], config['attention'])
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self.layers1 = nn.ModuleList([copy.deepcopy(encoder_layer1) for _ in range(len(self.layer_names1))])
        self.layers2 = nn.ModuleList([copy.deepcopy(encoder_layer2) for _ in range(len(self.layer_names2))])
        self.layers3 = nn.ModuleList([copy.deepcopy(encoder_layer3) for _ in range(len(self.layer_names3))])

        # if config['n_samples'] > 0:
        self.feat_aug = nn.ModuleList(
            [copy.deepcopy(encoder_layer2) for _ in range(2 * config['n_topic_transformers'])])
        self.n_iter_topic_transformer = config['n_topic_transformers']

        self.seed_tokens = nn.Parameter(torch.randn(config['n_topics'], config['d_model']))
        self.register_parameter('seed_tokens', self.seed_tokens)
        self.topic_drop = nn.Dropout1d(p=0.1)
        self.n_samples = config['n_samples']
        # self.avgpool = nn.AdaptiveAvgPool1d(160)
        # self.avgpool_1 = nn.AdaptiveAvgPool1d(96)
        self.norm_feat = nn.LayerNorm(576)
        # self.fea_down_t0 = conv1x1(config['d_model'] + config['d_model_fusion'] * 2, config['d_model_fusion'] * 2)
        # self.fea_down_t1 = conv1x1(config['d_model_fusion'] * 2, config['d_model_fusion'])
        # self.fea_down_t2 = conv1x1(config['d_model_fusion'], config['d_model'])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def sample_topic(self, prob_topics, topics, L):
        prob_topics0, prob_topics1 = prob_topics[:, :L], prob_topics[:, L:]
        topics0, topics1 = topics[:, :L], topics[:, L:]

        theta0 = F.normalize(prob_topics0.sum(dim=1), p=1, dim=-1)  # [N, K]
        theta1 = F.normalize(prob_topics1.sum(dim=1), p=1, dim=-1)
        theta = F.normalize(theta0 * theta1, p=1, dim=-1)
        if self.n_samples == 0:
            return None
        if self.training:
            sampled_inds = torch.multinomial(theta, self.n_samples)
            sampled_values = torch.gather(theta, dim=-1, index=sampled_inds)
        else:
            sampled_values, sampled_inds = torch.topk(theta, self.n_samples, dim=-1)
        sampled_topics0 = torch.gather(topics0, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics0.shape[1], 1))
        sampled_topics1 = torch.gather(topics1, dim=-1, index=sampled_inds.unsqueeze(1).repeat(1, topics1.shape[1], 1))
        return sampled_topics0, sampled_topics1

    def reduce_feat(self, feat, topick, N, C):
        len_topic = topick.sum(dim=-1).int()
        max_len = len_topic.max().item()
        selected_ids = topick.bool()
        resized_feat = torch.zeros((N, max_len, C), dtype=torch.float, device=feat.device)
        new_mask = torch.zeros_like(resized_feat[..., 0]).bool()
        for i in range(N):
            new_mask[i, :len_topic[i]] = True
        resized_feat[new_mask, :] = feat[selected_ids, :]
        return resized_feat, new_mask, selected_ids

    def forward(self, feats16_0, feats16_1, feats8_0, feats8_1, feats4_0, feats4_1, t64, t32, t16, data):
        mask = mask0 = mask1 = None
        N16, L16, C16 = feats16_0.shape
        N8, L8, C8 = feats8_0.shape
        N4, L4, C4 = feats4_0.shape
        NT64, LT64, CT64 = t64.shape
        NT32, LT32, CT32 = t32.shape
        NT16, LT16, CT16 = t16.shape
        H16 = int(math.sqrt(L16))
        H8 = int(math.sqrt(L8))
        H4 = int(math.sqrt(L4))
        HT64 = int(math.sqrt(LT64))
        HT32 = int(math.sqrt(LT32))
        HT16 = int(math.sqrt(LT16))

        kernel_size0 = H16 // HT32
        kernel_size1 = H4 // HT16
        kernel_size2 = H8 // HT64

        L = L8

        conf_matrix_16 = torch.einsum("nlc,nsc->nls", feats16_0, feats16_1) / C8 ** .5
        conf_matrix_16_f = conf_matrix_16.unsqueeze(1)
        # print(conf_matrix_16_f.shape)
        conf_matrix_16_f = F.interpolate(conf_matrix_16_f, scale_factor=4., mode='bilinear',
                                         align_corners=True)
        data.update({"conf_matrix_16_f": conf_matrix_16_f})
        # print(conf_matrix_16_f.shape)
        conf_matrix_16 = F.softmax(conf_matrix_16, 1) * F.softmax(conf_matrix_16, 2)
        conf_matrix_16_idx = torch.argmax(conf_matrix_16, -1)
        # print(conf_matrix_16_idx.shape)
        conf_matrix_16_idy = torch.argmax(conf_matrix_16, -2)
        feats16_1_ = feats16_1[:, conf_matrix_16_idx[-1]]
        feats16_t = torch.reshape((feats16_0 + feats16_1_) / 2,
                                  (feats16_0.shape[0], 20, 20, feats16_0.shape[2])).permute(
            0, 3, 1, 2)  # 2 80 80 256*2
        feats16_t = self.global_avg_pool10(feats16_t).permute(0, 2, 3, 1)  # 1,256,10,10
        feats16_t = torch.reshape(feats16_t, (NT32, LT32, feats16_t.shape[-1]))
        t32 = torch.concat((t32, feats16_t), -1)
        # print(t32.shape)
        feats16_0 = self.avgpool(torch.concat((feats16_0, conf_matrix_16_idx.unsqueeze(-1)), -1))
        feats16_1 = self.avgpool(torch.concat((feats16_1, conf_matrix_16_idy.unsqueeze(-1)), -1))

        # t32 = torch.reshape(t32, (t32.shape[0], 10, 10, t32.shape[2])).permute(0, 3, 1, 2)
        # t32 = self.global_avg_pool5(t32)  # 1,256,20,20
        t32 = self.avgpool(t32)
        t32 = t32.unsqueeze(2).squeeze(0)
        # print(t32.shape)
        # if mask0 is not None:
        #     mask = torch.cat((mask0, mask1), dim=-1)
        # else:
        #     mask = None

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'seed':
                t32s = []
                feats16_1 = feats16_1[:, conf_matrix_16_idx[-1]]
                feats16 = torch.concat((feats16_0, feats16_1), 0)
                feats16 = torch.reshape(feats16, (feats16.shape[0], 20, 20, feats16.shape[2])).permute(0, 3, 1, 2)
                feats16_u = F.unfold(feats16, kernel_size=(kernel_size0, kernel_size0), stride=kernel_size0,
                                     padding=0)
                feats16_u = rearrange(feats16_u, 'n (c ww) l -> n l ww c', ww=kernel_size0 ** 2)
                feats16_u = torch.concat((feats16_u[0], feats16_u[1]), -1)
                # print(feats16_u.shape)
                for ti, feati in zip(t32.squeeze(0), self.avgpool(feats16_u)):
                    # feats16_u=self.avgpool(feats16_u)
                    #     feat = torch.cat((feat0, feat1), dim=1)
                    # print(ti.shape)
                    # print(feati.shape)
                    ti = layer(ti.unsqueeze(0), feati.unsqueeze(0), None, None)
                    t32s.append(ti)
                t32 = torch.stack(t32s).squeeze(1)
                # print(t32.shape)
            elif name == 'feat':
                feats16_0 = layer(feats16_0, t32.permute(1, 0, 2), None, None)
                feats16_1 = layer(feats16_1, t32.permute(1, 0, 2), None, None)
        feats16 = torch.concat((feats16_0, feats16_1), 0)
        feats16 = torch.reshape(feats16, (feats16.shape[0], 20, 20, feats16.shape[2])).permute(0, 3, 1, 2)
        t32 = torch.reshape(t32, (NT32, HT32, HT32, t32.shape[-1])).permute(0, 3, 1, 2)
        # print(t32.shape)
        # print(feats16.shape)

        conf_matrix_4 = torch.einsum("nlc,nsc->nls", feats4_0, feats4_1) / C8 ** .5
        conf_matrix_4_f = conf_matrix_4.unsqueeze(1)
        conf_matrix_4_f = F.interpolate(conf_matrix_4_f, scale_factor=0.25, mode='bilinear',
                                        align_corners=True)
        data.update({"conf_matrix_4_f": conf_matrix_4_f})
        conf_matrix_4 = F.softmax(conf_matrix_4, 1) * F.softmax(conf_matrix_4, 2)
        conf_matrix_4_idx = torch.argmax(conf_matrix_4, -1)
        conf_matrix_4_idy = torch.argmax(conf_matrix_4, -2)
        feats4_1_ = feats4_1[:, conf_matrix_4_idx[-1]]
        feats4_t = torch.reshape((feats4_0 + feats4_1_) / 2, (feats4_0.shape[0], 80, 80, feats4_0.shape[2])).permute(
            0, 3, 1, 2)  # 2 80 80 256*2
        feats4_t = self.global_avg_pool20(feats4_t).permute(0, 2, 3, 1)  # 1,256,5,5
        feats4_t = torch.reshape(feats4_t, (NT16, LT16, feats4_t.shape[-1]))
        t16 = torch.concat((t16, feats4_t), -1)
        # feats4_0_ = feats4_0[:, conf_matrix_4_idy[-1]]
        # feats4_x = torch.concat((feats4_0_, feats4_1_), -1)
        # feats4_y = torch.concat((feats4_1_, feats4_0_), -1)
        feats4_0 = torch.concat((feats4_0, conf_matrix_4_idx.unsqueeze(-1)), -1)
        feats4_1 = torch.concat((feats4_1, conf_matrix_4_idy.unsqueeze(-1)), -1)
        t16 = self.avgpool1(t16)
        t16 = t16.unsqueeze(2).squeeze(0)

        for layer, name in zip(self.layers1, self.layer_names1):
            if name == 'seed':
                t16s = []
                feats4_1 = feats4_1[:, conf_matrix_4_idx[-1]]
                feats4 = torch.concat((feats4_0, feats4_1), 0)
                feats4 = torch.reshape(feats4, (feats4.shape[0], 80, 80, feats4.shape[2])).permute(0, 3, 1, 2)
                feats4_u = F.unfold(feats4, kernel_size=(kernel_size1, kernel_size1), stride=kernel_size1,
                                    padding=0)
                feats4_u = rearrange(feats4_u, 'n (c ww) l -> n l ww c', ww=kernel_size1 ** 2)
                feats4_u = torch.concat((feats4_u[0], feats4_u[1]), -1)

                # print(feats16_u.shape)
                for ti, feati in zip(t16, self.avgpool1(feats4_u)):
                    # feats16_u=self.avgpool(feats16_u)
                    #     feat = torch.cat((feat0, feat1), dim=1)
                    # print(ti.shape)
                    # print(feati.shape)
                    ti = layer(ti.unsqueeze(0), feati.unsqueeze(0), None, None)
                    t16s.append(ti)
                t16 = torch.stack(t16s).squeeze(1)

            elif name == 'feat':
                feats4_0 = layer(self.avgpool1(feats4_0), t16.permute(1, 0, 2), None, None)
                feats4_1 = layer(self.avgpool1(feats4_1), t16.permute(1, 0, 2), None, None)
        feats4 = torch.concat((feats4_0, feats4_1), 0)
        feats4 = torch.reshape(feats4, (feats4.shape[0], 80, 80, feats4.shape[2])).permute(0, 3, 1, 2)
        t16 = torch.reshape(t16, (NT16, HT16, HT16, t16.shape[-1])).permute(0, 3, 1, 2)
        # print(t16.shape)
        # print(feats4.shape)

        conf_matrix_8 = torch.einsum("nlc,nsc->nls", feats8_0, feats8_1) / C8 ** .5
        conf_matrix_8_f = conf_matrix_8.unsqueeze(1)

        data.update({"conf_matrix_8_f": conf_matrix_8_f})
        conf_matrix_8 = F.softmax(conf_matrix_8, 1) * F.softmax(conf_matrix_8, 2)
        conf_matrix_8_idx = torch.argmax(conf_matrix_8, -1)
        conf_matrix_8_idy = torch.argmax(conf_matrix_8, -2)
        feats8_1_ = feats8_1[:, conf_matrix_8_idx[-1]]
        feats8_t = torch.reshape((feats8_0 + feats8_1_) / 2, (feats8_0.shape[0], 40, 40, feats8_0.shape[2])).permute(
            0, 3, 1, 2)  # 2 80 80 256*2
        feats8_t = self.global_avg_pool5(feats8_t).permute(0, 2, 3, 1)  # 1,256,5,5
        feats8_t = torch.reshape(feats8_t, (NT64, LT64, feats8_t.shape[-1]))
        t64 = torch.concat((t64, feats8_t), -1)
        # feats4_0_ = feats4_0[:, conf_matrix_4_idy[-1]]
        # feats4_x = torch.concat((feats4_0_, feats4_1_), -1)
        # feats4_y = torch.concat((feats4_1_, feats4_0_), -1)
        feats8_0 = torch.concat((feats8_0, conf_matrix_8_idx.unsqueeze(-1)), -1)
        feats8_1 = torch.concat((feats8_1, conf_matrix_8_idy.unsqueeze(-1)), -1)
        t64 = self.avgpool2(t64)
        t64 = t64.unsqueeze(2).squeeze(0)
        # print(feats8_0.shape)
        # print(t64.shape)
        for layer, name in zip(self.layers2, self.layer_names2):
            if name == 'seed':
                t64s = []
                feats8_1 = feats8_1[:, conf_matrix_8_idx[-1]]
                feats8 = torch.concat((feats8_0, feats8_1), 0)
                feats8 = torch.reshape(feats8, (feats8.shape[0], 40, 40, feats8.shape[2])).permute(0, 3, 1, 2)
                feats8_u = F.unfold(feats8, kernel_size=(kernel_size2, kernel_size2), stride=kernel_size2,
                                    padding=0)
                feats8_u = rearrange(feats8_u, 'n (c ww) l -> n l ww c', ww=kernel_size2 ** 2)
                feats8_u = torch.concat((feats8_u[0], feats8_u[1]), -1)
                # print(feats16_u.shape)
                for ti, feati in zip(t64, self.avgpool2(feats8_u)):
                    # feats16_u=self.avgpool(feats16_u)
                    #     feat = torch.cat((feat0, feat1), dim=1)
                    # print(ti.shape)
                    # print(feati.shape)
                    ti = layer(ti.unsqueeze(0), feati.unsqueeze(0), None, None)
                    t64s.append(ti)
                t64 = torch.stack(t64s).squeeze(1)
                # print(t32.shape)
            elif name == 'feat':
                feats8_0 = layer(self.avgpool2(feats8_0), t64.permute(1, 0, 2), None, None)
                feats8_1 = layer(self.avgpool2(feats8_1), t64.permute(1, 0, 2), None, None)
        feats8 = torch.concat((feats8_0, feats8_1), 0)
        feats8 = torch.reshape(feats8, (feats8.shape[0], 40, 40, feats8.shape[2])).permute(0, 3, 1, 2)
        t64 = torch.reshape(t64, (NT64, HT64, HT64, t64.shape[-1])).permute(0, 3, 1, 2)
        # print(t64.shape)
        # print(t32.shape)
        # print(t16.shape)
        # print(feats16.shape)
        # print(feats8.shape)
        # print(feats4.shape)
        # torch.Size([1, 5, 5, 192])
        # torch.Size([1, 10, 10, 256])
        # torch.Size([1, 20, 20, 128])
        # torch.Size([2, 256, 20, 20])
        # torch.Size([2, 192, 40, 40])
        # torch.Size([2, 128, 80, 80])
        t16 = self.global_avg_pool5(t16)  # 1,256,5,5
        t32 = self.global_avg_pool5(t32)  # 1,256,5,5
        feats16 = F.interpolate(feats16, scale_factor=2, mode='bilinear', align_corners=True)
        feats4 = F.interpolate(feats4, scale_factor=0.5, mode='bilinear', align_corners=True)
        t64 = torch.reshape(torch.concat((t64, t32, t16), 1).permute(0, 2, 3, 1), (NT64, LT64, -1)).permute(1, 0,
                                                                                                            2) / C8 ** 0.5
        t64 = self.norm_feat(t64)
        feats8 = torch.reshape(torch.concat((feats8, feats16, feats4), 1).permute(0, 2, 3, 1),
                               (N8 * 2, L8, -1)) / C8 ** 0.5
        feats8_0 = feats8[0].unsqueeze(0)
        feats8_1 = feats8[1].unsqueeze(0)
        # print(t64.shape)
        # print(feats8.shape)
        # print(feats8_0.shape)
        # print(feats8_1.shape)

        for layer, name in zip(self.layers3, self.layer_names3):
            if name == 'seed':
                t64s = []
                feats8_1 = feats8_1[:, conf_matrix_8_idx[-1]]
                feats8 = torch.concat((feats8_0, feats8_1), 0)
                feats8 = torch.reshape(feats8, (feats8.shape[0], 40, 40, feats8.shape[2])).permute(0, 3, 1, 2)
                feats8_u = F.unfold(feats8, kernel_size=(kernel_size2, kernel_size2), stride=kernel_size2,
                                    padding=0)
                feats8_u = rearrange(feats8_u, 'n (c ww) l -> n l ww c', ww=kernel_size2 ** 2)
                feats8_u = torch.concat((feats8_u[0], feats8_u[1]), -1)
                # print(feats16_u.shape)
                for ti, feati in zip(self.avgpool2(t64), self.avgpool2(feats8_u)):
                    # feats16_u=self.avgpool(feats16_u)
                    #     feat = torch.cat((feat0, feat1), dim=1)
                    # print(ti.shape)
                    # print(feati.shape)
                    # print(ti.unsqueeze(0))
                    # print(feati.unsqueeze(0))
                    ti = layer(ti.unsqueeze(0), feati.unsqueeze(0), None, None)
                    t64s.append(ti)
                t64 = torch.stack(t64s).squeeze(1)
                # print(t32.shape)
            elif name == 'feat':
                feats8_0 = layer(self.avgpool2(feats8_0), t64.permute(1, 0, 2), None, None)
                feats8_1 = layer(self.avgpool2(feats8_1), t64.permute(1, 0, 2), None, None)
        feats8 = torch.concat((feats8_0, feats8_1), 1)
        t64 = t64.permute(1, 0, 2)
        # feats8 = torch.reshape(feats8, (feats8.shape[0], 40, 40, feats8.shape[2])).permute(0, 3, 1, 2)
        # feats8 = torch.reshape(feats8, (feats8.shape[0], 40, 40, feats8.shape[2])).permute(0, 3, 1, 2)
        # t64 = torch.reshape(t64, (NT64, HT64, HT64, t64.shape[-1])).permute(0, 3, 1, 2)
        # t64 = torch.reshape(t64, (NT64, HT64, HT64, t64.shape[-1])).permute(0, 3, 1, 2)
        # print(feats8.shape)
        # print(t64.shape)
        seeds = t64
        feat = feats8
        dmatrix = torch.einsum("nmd,nkd->nmk", feat, seeds) / C8 ** .5
        prob_topics = F.softmax(dmatrix, dim=-1)

        feat_topics = torch.zeros_like(dmatrix).scatter_(-1, torch.argmax(dmatrix, dim=-1, keepdim=True), 1.0)

        if mask is not None:
            feat_topics = feat_topics * mask.unsqueeze(-1)
            prob_topics = prob_topics * mask.unsqueeze(-1)

        sampled_topics = self.sample_topic(prob_topics.detach(), feat_topics, L)
        if sampled_topics is not None:
            updated_feat0, updated_feat1 = torch.zeros_like(feats8_0), torch.zeros_like(feats8_1)
            s_topics0, s_topics1 = sampled_topics
            for k in range(s_topics0.shape[-1]):
                topick0, topick1 = s_topics0[..., k], s_topics1[..., k]  # [N, L+S]
                if (topick0.sum() > 0) and (topick1.sum() > 0):
                    new_feat0, new_mask0, selected_ids0 = self.reduce_feat(feats8_0, topick0, N8, C8 // 4 * 3)
                    new_feat1, new_mask1, selected_ids1 = self.reduce_feat(feats8_1, topick1, N8, C8 // 4 * 3)
                    for idt in range(self.n_iter_topic_transformer):
                        new_feat0 = self.feat_aug[idt * 2](new_feat0, new_feat0, new_mask0, new_mask0)
                        new_feat1 = self.feat_aug[idt * 2](new_feat1, new_feat1, new_mask1, new_mask1)
                        new_feat0 = self.feat_aug[idt * 2 + 1](new_feat0, new_feat1, new_mask0, new_mask1)
                        new_feat1 = self.feat_aug[idt * 2 + 1](new_feat1, new_feat0, new_mask1, new_mask0)
                    updated_feat0[selected_ids0, :] = new_feat0[new_mask0, :]
                    updated_feat1[selected_ids1, :] = new_feat1[new_mask1, :]

            feats8_0 = (1 - s_topics0.sum(dim=-1, keepdim=True)) * feats8_0 + updated_feat0
            feats8_1 = (1 - s_topics1.sum(dim=-1, keepdim=True)) * feats8_1 + updated_feat1
        else:
            for idt in range(self.n_iter_topic_transformer * 2):
                feats8_0 = self.feat_aug[idt](feats8_0, seeds, mask0, None)
                feats8_1 = self.feat_aug[idt](feats8_1, seeds, mask1, None)

        if self.training:
            # topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:])
            # topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:]) / C ** .5
            # print(topic_matrix)
            topic_matrix_img0 = feat_topics[:, :L]
            Nt, Ct, Ht, Wt = topic_matrix_img0.shape[0], topic_matrix_img0.shape[2], int(math.sqrt(
                topic_matrix_img0.shape[1])), int(math.sqrt(topic_matrix_img0.shape[1]))

            topic_matrix_img0 = torch.reshape(topic_matrix_img0, (Nt, Ct, Ht, Wt))

            topic_matrix_img0_unfold = F.unfold(topic_matrix_img0, kernel_size=(3, 3), stride=1, padding=1)
            topic_matrix_img0_unfold = rearrange(topic_matrix_img0_unfold, 'n (c ww) l -> n l ww c',
                                                 ww=3 ** 2)  # 2*144*9*256
            topic_matrix_img0_unfold = torch.mean(torch.transpose(topic_matrix_img0_unfold, -1, -2), -1)
            topic_matrix_img0_unfold_mask = topic_matrix_img0_unfold > 0.5
            # topic_matrix_img0_unfold_mask = topic_matrix_img0_unfold > 0
            topic_matrix_img0_unfold_mask = topic_matrix_img0_unfold_mask.byte()
            topic_matrix_img0_unfold_mask = torch.reshape(topic_matrix_img0_unfold_mask, (Nt, Ht * Wt, Ct))
            topic_matrix_img0 = torch.reshape(topic_matrix_img0, (Nt, Ht * Wt, Ct))

            topic_matrix_img0 = topic_matrix_img0 * topic_matrix_img0_unfold_mask

            topic_matrix_img1 = feat_topics[:, L:]

            topic_matrix_img1 = torch.reshape(topic_matrix_img1, (Nt, Ct, Ht, Wt))

            topic_matrix_img1_unfold = F.unfold(topic_matrix_img1, kernel_size=(3, 3), stride=1, padding=1)
            topic_matrix_img1_unfold = rearrange(topic_matrix_img1_unfold, 'n (c ww) l -> n l ww c',
                                                 ww=3 ** 2)  # 2*144*9*256
            topic_matrix_img1_unfold = torch.mean(torch.transpose(topic_matrix_img1_unfold, -1, -2), -1)
            topic_matrix_img1_unfold_mask = topic_matrix_img1_unfold > 0.5
            # topic_matrix_img1_unfold_mask = topic_matrix_img1_unfold > 0
            topic_matrix_img1_unfold_mask = topic_matrix_img1_unfold_mask.byte()
            topic_matrix_img1 = torch.reshape(topic_matrix_img1, (Nt, Ht * Wt, Ct))
            topic_matrix_img1_unfold_mask = torch.reshape(topic_matrix_img1_unfold_mask, (Nt, Ht * Wt, Ct))
            topic_matrix_img1 = topic_matrix_img1 * topic_matrix_img1_unfold_mask
            topic_matrix = torch.einsum("nlk,nsk->nls", prob_topics[:, :L], prob_topics[:, L:])
            # print(topic_matrix)
            topic_matrix_match = {"img0": topic_matrix_img0, "img1": topic_matrix_img1}

        else:
            topic_matrix = {"img0": feat_topics[:, :L], "img1": feat_topics[:, L:]}
            topic_matrix_match = {"img0": feat_topics[:, :L], "img1": feat_topics[:, L:]}

        return feats8_0, feats8_1, topic_matrix, topic_matrix_match, seeds


# class FineNetwork(nn.Module):

#     def __init__(self, config, add_detector=True):
#         super(FineNetwork, self).__init__()

#         self.config = config
#         self.d_model = config['d_model']
#         self.nhead = config['nhead']
#         self.layer_names = config['layer_names']
#         self.n_mlp_mixer_blocks = config["n_mlp_mixer_blocks"]
#         self.encoder_layers = nn.ModuleList([MLPMixerEncoderLayer(config["n_feats"] * 2, self.d_model)
#                                              for _ in range(self.n_mlp_mixer_blocks)])
#         self.detector = None
#         if add_detector:
#             self.detector = nn.Sequential(MLPMixerEncoderLayer(config["n_feats"], self.d_model),
#                                           nn.Linear(self.d_model, 1))

#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, feat0, feat1, mask0=None, mask1=None):
#         """
#         Args:
#             feat0 (torch.Tensor): [N, L, C]
#             feat1 (torch.Tensor): [N, S, C]
#             mask0 (torch.Tensor): [N, L] (optional)
#             mask1 (torch.Tensor): [N, S] (optional)
#         """

#         assert self.d_model == feat0.shape[2], "the feature number of src and transformer must be equal"

#         feat = torch.cat((feat0, feat1), dim=1)
#         for idx in range(self.n_mlp_mixer_blocks):
#             feat = self.encoder_layers[idx](feat)
#         feat0, feat1 = feat[:, :feat0.shape[1]], feat[:, feat0.shape[1]:]
#         score_map0 = None
#         if self.detector is not None:
#             score_map0 = self.detector(feat0).squeeze(-1)

#         return feat0, feat1, score_map0


# class FeatureFusion(nn.Module):

#     def __init__(self, config):
#         super(FeatureFusion, self).__init__()

#         self.config = config
#         self.d_model = config['d_model']
#         self.d_model_fusion = config['d_model']
#         self.nhead = config['nhead']
#         self.avgpool = nn.AdaptiveAvgPool1d(160)
#         self.avgpool_1 = nn.AdaptiveAvgPool1d(96)
#         self.norm_feat = nn.LayerNorm(self.d_model)
#         # self.down_dim0 = conv1x1(self.d_model_fusion * 2, self.d_model * 3 // 2)
#         # self.down_dim1 = conv1x1(self.d_model_fusion * 3 // 2, self.d_model)
#         # self.down_dim = conv1x1(self.d_model_fusion, self.d_model - 96)
#         self.head = 4
#         self.thr = config['thr']
#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, feat_s0, feat_s1, data=None):
#         H = W = int(math.sqrt(feat_s0.shape[1]))
#         N, C, K = feat_s0.shape[0], feat_s0.shape[2], self.config['n_topics']

#         assert self.d_model_fusion == C, "the feature number of src and transformer must be equal"

#         feat_s0 = torch.reshape(feat_s0, (N, H * W, C))
#         feat_s1 = torch.reshape(feat_s1, (N, H * W, C))

#         conf_matrix = torch.einsum("nlc,nsc->nls", feat_s0, feat_s1) / self.d_model ** .5  # (C * temperature)

#         feat_s0 = torch.reshape(feat_s0, (N, H * W, C))
#         feat_s1 = torch.reshape(feat_s1, (N, H * W, C))
#         conf_matrix = conf_matrix

#         data["conf_matrix"] = conf_matrix
#         conf_matrix = F.softmax(conf_matrix, 1) * F.softmax(conf_matrix, 2)
#         conf_mask = conf_matrix > self.thr
#         conf_mask = conf_mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
#                     * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
#         conf_mask = conf_mask.float()  # 2*144*144

#         feat_s0 = torch.reshape(feat_s0, (N, C, H, W))
#         feat_s1 = torch.reshape(feat_s1, (N, C, H, W))

#         feat_s0_unfold = F.unfold(feat_s0, kernel_size=(3, 3), stride=1, padding=1)
#         feat_s0_unfold = rearrange(feat_s0_unfold, 'n (c ww) l -> n l ww c', ww=3 ** 2)  # 2*144*9*256

#         feat_s0 = torch.reshape(feat_s0, (N, H * W, C))
#         feat_s0_sem = torch.einsum("nlc,nlwc->nlw", feat_s0, feat_s0_unfold) / feat_s0_unfold.shape[-1]  # 2*144*9

#         feat_s1_unfold = F.unfold(feat_s1, kernel_size=(3, 3), stride=1, padding=1)
#         feat_s1_unfold = rearrange(feat_s1_unfold, 'n (c ww) l -> n l ww c', ww=3 ** 2)
#         feat_s1 = torch.reshape(feat_s1, (N, H * W, C))
#         feat_s1_sem = torch.einsum("nlc,nlwc->nlw", feat_s1, feat_s1_unfold) / feat_s1_unfold.shape[-1]  # 2*144*9

#         feat_s0_fea = torch.einsum("nlwc,nlw->nlc", feat_s0_unfold, feat_s0_sem)  # 2*144*256
#         feat_s1_fea = torch.einsum("nlwc,nlw->nlc", feat_s1_unfold, feat_s1_sem)  # 2*144*256
#         feat_s0_sem_fea = torch.einsum("nlc,ncd->nld", conf_mask, feat_s1_fea) / conf_mask.shape[1]

#         conf_mask_trans = torch.transpose(conf_mask, -2, -1)
#         feat_s1_sem_fea = torch.einsum("nlc,ncd->nld", conf_mask_trans, feat_s0_fea) / conf_mask_trans.shape[1]

#         conf_matrix_fusion = torch.einsum("nlc,nsc->nls", feat_s0_sem_fea,
#                                           feat_s1_sem_fea) / self.d_model ** .5  # (C * temperature)
#         data['conf_matrix_fusion'] = conf_matrix_fusion
#         feat_s0 = self.avgpool(feat_s0)
#         feat_s0_sem_fea = self.avgpool_1(feat_s0_sem_fea)
#         feat_s1 = self.avgpool(feat_s1)
#         feat_s1_sem_fea = self.avgpool_1(feat_s1_sem_fea)
#         feat_s0 = torch.concat((feat_s0, feat_s0_sem_fea), -1)
#         feat_s1 = torch.concat((feat_s1, feat_s1_sem_fea), -1)
#         feat_s = torch.concat((feat_s0, feat_s1), 0)
#         feat_s = self.norm_feat(feat_s)
#         # feat_s = torch.reshape(feat_s, (N * 2, C * 2, H, W))
#         # feat_s = self.down_dim0(feat_s)
#         # feat_s = self.down_dim1(feat_s)
#         # feat_s = torch.reshape(feat_s, (N * 2, H * W, self.d_model))

#         return feat_s[:feat_s0.shape[0]], feat_s[feat_s0.shape[0]:]


class Relator_Fusion(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, config):
        super(Relator_Fusion, self).__init__()

        self.config = config
        self.d_model = config['d_model']
        self.d_model_relator = config['d_model']
        self.up_dim_ini = config['relator_dim_ini']
        self.avgpool = nn.AdaptiveAvgPool1d(160)
        self.avgpool_1 = nn.AdaptiveAvgPool1d(96)
        # self.down_dim0 = conv1x1(self.d_model * 2, self.d_model * 3 // 2)
        # self.down_dim1 = conv1x1(self.d_model * 3 // 2, self.d_model)
        # self.layer_names = config['layer_names_relator']
        # encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
        # self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self.norm_feat = nn.LayerNorm(self.d_model)
        self.head = 4
        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if 'temp' in name or 'sample_offset' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, data=None):
        """
        Args:
            feat0 (torch.Tensor): [N, C, H, W]
            feat1 (torch.Tensor): [N, C, H, W]
            pos1,pos2:  [N, C, H, W]
        Outputs:
            feat0: [N,-1,C]
            feat1: [N,-1,C]
            flow_list: [L,N,H,W,4]*1(2)
        """
        # print(feat0.shape)
        # print(feat1.shape)
        relator = torch.einsum("nlc,nsc->nls", feat0, feat1) / self.d_model ** .5  # B,L,S
        relator = torch.unsqueeze(relator, 1)
        conf_matrix_fusion = data['conf_matrix_fusion']
        conf_matrix_fusion = torch.unsqueeze(conf_matrix_fusion, 1)
        conf_matrix = data['conf_matrix']
        conf_matrix = torch.unsqueeze(conf_matrix, 1)
        relators = torch.concat((conf_matrix, relator, conf_matrix_fusion), 1)
        data["conf_matrix_relator"] = torch.squeeze(torch.mean(relators, 1), 1)
        relator_trans = torch.transpose(relators, -1, -2)
        relators = torch.concat((relators, relator_trans), 0)
        relators = torch.reshape(relators,
                                 (relators.shape[0], relators.shape[2], relators.shape[3] * relators.shape[1]))
        # print(relators.shape)
        # relators = self.relators_cnn(relators)
        relators = self.avgpool_1(relators)
        # print(relators.shape)

        # relators = torch.transpose(relators, -1, -2)
        feat = torch.concat((feat0, feat1), 0)
        # print(feat.shape)
        feat = self.avgpool(feat)
        # for layer, name in zip(self.layers, self.layer_names):
        #     if name == 'feat':
        #         feat = layer(feat, relators, None)
        feat = torch.concat((feat, relators), -1)
        # feat = self.down_dim0(torch.reshape(feat, (feat.shape[0], feat.shape[2], int(math.sqrt(
        #     feat.shape[1])), int(math.sqrt(feat.shape[1])))))
        # feat = self.down_dim1(feat)
        # feat = self.avgpool(feat)
        # feat = torch.reshape(feat, (feat.shape[0], feat.shape[2] * feat.shape[3], feat.shape[1]))
        feat = self.norm_feat(feat)

        feat0 = feat[:feat0.shape[0]]
        feat1 = feat[feat0.shape[0]:]

        return feat0, feat1
