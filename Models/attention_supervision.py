import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionSupervision(nn.Module):
    """计算分割token与图像token之间的注意力图，并使用KL散度监督"""
    
    def __init__(self):
        super(AttentionSupervision, self).__init__()
    
    def compute_attention_map(self, attention_maps, seg_token_indices):
        """
        计算分割token与图像token之间的注意力图
        
        参数:
            attention_maps: 形状为 [L, H, Q, P] 的注意力图，其中:
                L是transformer层数
                H是注意力头数
                Q是查询序列长度
                P是键/值序列长度
            seg_token_indices: 分割token在序列中的索引
            
        返回:
            形状为 [L, H, Q, P] 的注意力图
        """
        # 提取分割token的注意力图
        seg_attention = []
        for layer_idx in range(len(attention_maps)):
            layer_attention = attention_maps[layer_idx]
            # 提取分割token的注意力权重
            seg_attn = layer_attention[:, seg_token_indices, :]
            seg_attention.append(seg_attn)
        
        # 将所有层的注意力图堆叠起来
        seg_attention = torch.stack(seg_attention, dim=0)  # [L, H, len(seg_indices), P]
        
        # 对注意力头维度求平均
        avg_attention = torch.mean(seg_attention, dim=1)  # [L, len(seg_indices), P]
        
        return avg_attention
    
    def compute_kl_loss(self, attention_map, ground_truth_mask):
        """
        计算注意力图与真实分割掩码之间的KL散度损失
        
        参数:
            attention_map: 形状为 [L, Q, P] 的注意力图
            ground_truth_mask: 形状为 [Q, P] 的真实分割掩码
            
        返回:
            KL散度损失
        """
        # 确保ground_truth_mask是概率分布
        if ground_truth_mask.sum() > 0:
            ground_truth_mask = ground_truth_mask / ground_truth_mask.sum(dim=-1, keepdim=True)
        
        # 计算每一层的KL散度损失
        kl_losses = []
        for layer_idx in range(attention_map.shape[0]):
            layer_attn = attention_map[layer_idx]  # [Q, P]
            
            # 确保注意力图是概率分布
            layer_attn = F.softmax(layer_attn, dim=-1)
            
            # 计算KL散度
            kl_div = F.kl_div(
                layer_attn.log(),
                ground_truth_mask,
                reduction='batchmean'
            )
            kl_losses.append(kl_div)
        
        # 对所有层的损失求平均
        kl_loss = torch.mean(torch.stack(kl_losses))
        return kl_loss