import errno
import os
import sys

from torch.cuda.amp import GradScaler

from dassl.engine import TRAINER_REGISTRY
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip

from trainers.baseda import *
from utils.clip_part import *
from utils.templates import CUSTOM_TEMPLATES

from openTSNE import TSNE
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain



class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.text_encoder = Simple_TextEncoder(clip_model)

        self.image_encoder = clip_model.visual
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts = clip.tokenize(prompts)
    
    def forward(self, image):
        text_features = self.text_encoder(self.tokenized_prompts.to(self.logit_scale.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t().cuda(image_features.device)

        return logits


@TRAINER_REGISTRY.register()
class CLIP_ZS(BaseDA):
    """
    ZS: Zero-Shot CLIP
    """  
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.CLIP.PREC == "fp32" or cfg.TRAINER.CLIP.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print("# params: {:,}".format(0))

        self.model.to(self.device)

        # transform the epoch to step schedule
        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError

        # no loss
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("CLIP_model", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.CLIP.PREC == "amp" else None

        self.T_SNE_combined()
        sys.exit()

    @torch.no_grad()
    def T_SNE_combined(self):
        self.set_model_mode("eval")

        all_embeddings = []
        all_labels = []

        combined_loader = chain(self.train_loader_x, self.train_loader_u)

        for batch_idx, batch in enumerate(combined_loader):
            input, label = self.parse_batch_test(batch)

            image_features = self.model.image_encoder(input.type(self.model.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            all_embeddings.append(image_features.cpu().numpy())
            if batch_idx < len(self.train_loader_x):
                all_labels.extend([0] * len(label))
            else:
                all_labels.extend([1] * len(label))

        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.array(all_labels)

        tsne = TSNE(perplexity=50, metric="euclidean", random_state=42)
        embeddings = tsne.fit(all_embeddings)

        source_mask = all_labels == 0
        target_mask = all_labels == 1

        # 创建散点图
        plt.figure(figsize=(10, 8))

        plt.scatter(embeddings[source_mask, 0], embeddings[source_mask, 1], color='blue', marker='o', s=8,
                    label='Source domain', alpha=0.8)
        plt.scatter(embeddings[target_mask, 0], embeddings[target_mask, 1], color='red', marker='o', s=8,
                    label='Target domain', alpha=0.8)

        # 添加图例和其他装饰
        plt.xticks(())
        plt.yticks(())

        out_dir = "/data/dxw/PMCC/tsne/CLIP/Office31"

        plt.title('CLIP' + ' (' + str(self.domains).upper() + ')', fontdict={"family": "Times New Roman", "size": 64})

        plt.savefig(out_dir + '/CLIP-' + str(self.domains).upper() + '.pdf')

    def train(self):
        self.before_train()
        self.after_train()
    
        