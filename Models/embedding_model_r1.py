import ml_collections  # 导入ml_collections库，用于管理配置
import torch  # 导入PyTorch核心库
import torch.nn as nn  # 导入PyTorch的神经网络模块
from typing import Dict, List, Optional, Tuple, Union
from .common_arch import AttnPooler, LayerNorm, LayerNormFp32, MoEProjection
import torch_npu


class EmbeddingModel(nn.Module):
    def __init__(self, config: ml_collections.ConfigDict):
        """
        初始化 EmbeddingModel 模型。
        """
        super(EmbeddingModel, self).__init__()

        # 使用LHRS的连接层
        if config.adjust_norm:
            norm_layer = (
                LayerNormFp32 if config.dtype in ("float16", "bfloat16") else LayerNorm
            )
        else:
            norm_layer = LayerNorm

        moe_cfg = getattr(config, "moe_proj", ml_collections.ConfigDict())

        # 🌟 修复 1：实例化时接入任务感知参数
        self.projection = MoEProjection(
            num_experts=int(moe_cfg.get("num_experts", 4)),
            num_query=config.rgb_vision.attn_pooler.num_query,
            num_layers=config.rgb_vision.attn_pooler.num_layers,
            num_attention_heads=config.rgb_vision.attn_pooler.num_attn_heads,
            encoder_hidden_size=getattr(config, "alignment_dim", 768),
            hidden_size=getattr(config, "alignment_dim", 768),
            output_size=config.text.hidden_size,
            # ====== 新增的任务感知参数 ======
            num_tasks=int(moe_cfg.get("num_tasks", 3)),  # 从 config 读取任务总数
            task_dim=int(moe_cfg.get("task_dim", 256)),  # 任务 Embedding 维度
            # ==============================
            norm_layer=norm_layer,
            checkpoint=getattr(config, "use_checkpoint", False),
            top_k=int(moe_cfg.get("top_k", 2)),
        )

    def forward(self, data: Dict, image_embedding: torch.Tensor):
        """
        前向传播，生成多模态嵌入。
        """
        # 🌟 修复 2：从 data 字典中安全提取 task_ids
        # 防御性编程：如果 DataLoader 没有传 task_ids (比如纯预训练阶段)，默认全部给 0
        batch_size = image_embedding.shape[0]
        device = image_embedding.device

        if "task_ids" in data:
            task_ids = data["task_ids"].to(device)
        else:
            # 兼容老的预训练 DataLoader：全部视为任务 0 (例如：图文对齐任务)
            task_ids = torch.zeros(batch_size, dtype=torch.long, device=device)

        # 🌟 传入 task_ids，激活双重门控路由！
        projected_multimodal_embedding = self.projection(image_embedding, task_ids)
        return projected_multimodal_embedding

    def encode_test(self, image_embedding: torch.Tensor, task_ids: Optional[torch.Tensor] = None):
        """
        纯图像编码测试。
        """
        # 推理阶段也需要 task_ids，如果不传，默认设为 0
        if task_ids is None:
            batch_size = image_embedding.shape[0]
            task_ids = torch.zeros(batch_size, dtype=torch.long, device=image_embedding.device)

        projected_image_embedding = self.projection(image_embedding, task_ids)
        return projected_image_embedding

    # 🌟 修复 3：对外暴露辅助损失和统计信息的接口
    # 这样外层的 CoastGPT 就能直接调用 self.multimodal.get_aux_loss() 了
    def get_aux_loss(self) -> torch.Tensor:
        if hasattr(self.projection, "get_aux_loss"):
            return self.projection.get_aux_loss()
        return torch.tensor(0.0)

    def get_gate_stats(self) -> Dict[str, torch.Tensor]:
        if hasattr(self.projection, "get_gate_stats"):
            return self.projection.get_gate_stats()
        return {}