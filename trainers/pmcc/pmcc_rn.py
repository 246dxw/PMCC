import sys

import numpy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.baseda import *
from utils.clip_part import *
from utils.templates import CUSTOM_TEMPLATES
from utils.single_ift_block import *

from openTSNE import TSNE
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

_tokenizer = _Tokenizer()


def load_clip_to_cpu_teacher(cfg):
    backbone_name = "ViT-B/16"
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, cfg.MODEL.BACKBONE.PATH)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class PromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.PMCC.N_CTX
        ctx_init = cfg.TRAINER.PMCC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]   # text encoder hidden size(512)
        self.dim = clip_model.text_projection.shape[1]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.tp = cfg.TRAINER.PMCC.TP
        self.vp = cfg.TRAINER.PMCC.VP
        self.t_deep = cfg.TRAINER.PMCC.T_DEEP
        self.v_deep = cfg.TRAINER.PMCC.V_DEEP
        self.deep_share = cfg.TRAINER.PMCC.DEEP_SHARED
        self.share_layer = cfg.TRAINER.PMCC.SHARE_LAYER
        self.num_tokens = cfg.TRAINER.PMCC.NUM_TOKENS    # number of prompted tokens
        self.deep_layer = cfg.TRAINER.PMCC.DEEP_LAYERS # num of layer has prompt ([1,3]: 1~3 layer has)
        self.location = cfg.TRAINER.PMCC.LOCATION
        self.prompt_dropout = nn.Dropout(cfg.TRAINER.PMCC.DROPOUT)
        self.num_layer = cfg.MODEL.NUM_LAYER
        self.hidden_size = cfg.MODEL.HIDDEN_SIZE    # visual encoder hiden size(768)

        self.ctx = None
        if self.tp:
            if ctx_init and n_ctx <= 4:   # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                self.ctx = nn.Parameter(ctx_vectors)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = ctx_init
            else:
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
            self.ctx = nn.Parameter(ctx_vectors)

        self.deep_ctx = None
        if self.t_deep:
            if self.deep_layer == None:
                deep_ctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, ctx_dim)
            else:
                deep_ctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] + 1, self.num_tokens, ctx_dim)
            nn.init.normal_(deep_ctx_vectors, std=0.02)
            self.deep_ctx = nn.Parameter(deep_ctx_vectors)

        print('Prompt design: Prompt-affinity Multi-modal Class Centroids for UDA')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of PMCC context words (tokens): {n_ctx}")

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

        self.attn_block = IFT_Module(clip_model, beta_s=0.1)
        self.K = 5
        self.dim = clip_model.text_projection.shape[1]

        domains = cfg.DOMAINS
        if cfg.DATASET.NAME == "OfficeHome":
            DOMAINS = {'a': "art", 'c': "clipart", 'p': "product", 'r': "real_world"}
        elif cfg.DATASET.NAME == "VisDA17":
            DOMAINS = {'s': "synthetic", 'r': "real"}
        elif cfg.DATASET.NAME == "Office31":
            DOMAINS = {'a': "amazon", 'w': "webcam", 'd': "dslr"}
        source_domain, target_domain = domains.split('-')[0], domains.split('-')[1]

        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'RN50':
            self.source_feat_bank = torch.load(
                "/data/dxw/PMCC/assets/" + cfg.DATASET.NAME + "/" + DOMAINS[source_domain] + "Centroid_RN50.pt")
        else:
            self.source_feat_bank = torch.load(
                "/data/dxw/PMCC/assets/" + cfg.DATASET.NAME + "/" + DOMAINS[source_domain] + "Centroid_RN101.pt")



    def forward(self):

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)   # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)


        return prompts, self.deep_ctx

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
    elif classname.find('BatchNorm') != -1:
        m.bias.requires_grad_(False)
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model, clip_model_vit):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.K = self.prompt_learner.K
        self.dim = clip_model.text_projection.shape[1]

        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)

        self.n_cls = len(classnames)
        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = torch.device("cuda:{}".format(cfg.GPU))

        self.classifier_layer = nn.Sequential(
            nn.LayerNorm(self.dim, eps=1e-6),
            nn.Linear(self.dim, self.n_cls, bias=False)).half()
        self.classifier_layer.apply(weights_init_classifier).to(self.device)

        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'RN50':
            self.text_features_u = torch.load(
                "/data/dxw/PMCC/assets/RN50/text_features" + cfg.DATASET.NAME + ".pt").half().to(self.device)
        else:
            self.text_features_u = torch.load(
                "/data/dxw/PMCC/assets/RN101/text_features" + cfg.DATASET.NAME + ".pt").half().to(self.device)

        self.image_encoder_vit = ImageEncoder_Trans(cfg, clip_model_vit)
        self.text_features_u_vit = torch.load(
            "/data/dxw/PMCC/assets/VIT/text_features" + cfg.DATASET.NAME + ".pt").half().to(self.device)

        source_bank = torch.mean(self.prompt_learner.source_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1).to(self.device)
        self.bank = 0.5 * source_bank + 0.5 * self.text_features_u

        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_u = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts_u = clip.tokenize(prompts_u)

        self.confi = cfg.CONFI
        self.epoch = cfg.EPOCH
        self.warm_up = cfg.WARM_UP

    def forward(self, image_x, label=None, image_u=None, epoch=None, train=False):

        prompts, deep_ctx = self.prompt_learner()
        logit_scale = self.logit_scale.exp()

        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_ctx)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features_x = self.image_encoder(image_x.type(self.dtype))
        image_features_x = image_features_x / image_features_x.norm(dim=-1, keepdim=True)

        logits_x = logit_scale * image_features_x @ text_features.t()
        logits_c_x = self.prompt_learner.attn_block(text_features, image_features_x, self.bank)
        logits_a_x = self.classifier_layer(image_features_x)

        if not train:
            return 0.7 * logits_x + 0.3 * logits_a_x + 0.2 * logits_c_x

        F_u = self.image_encoder_vit(image_u.type(self.dtype), None, None)
        F_u = F_u / F_u.norm(dim=-1, keepdim=True)

        logits_clip = logit_scale * F_u @ self.text_features_u_vit.t()

        image_features_u = self.image_encoder(image_u.type(self.dtype))
        image_features_u = image_features_u / image_features_u.norm(dim=-1, keepdim=True)

        logits_u = logit_scale * image_features_u @ text_features.t()
        logits_c_u = self.prompt_learner.attn_block(text_features, image_features_u, self.bank)
        logits_a_u = self.classifier_layer(image_features_u)

        if epoch == None or epoch <= self.epoch:
            pseudo_label = torch.softmax(logits_clip, dim=-1)
        else:
            pseudo_label = torch.softmax(logits_u, dim=-1)
        max_probs, label_p = torch.max(pseudo_label, dim=-1)

        mask = max_probs.ge(self.confi).float()

        if mask.sum() == 0 or self.warm_up > epoch:
            loss_u = torch.tensor(0.)
        else:
            loss_u = (F.cross_entropy(logits_u, label_p, reduction="none") * mask).sum() / mask.sum()
            loss_u += (F.cross_entropy(logits_a_u, label_p, reduction="none") * mask).sum() / mask.sum()
            loss_u += (F.cross_entropy(logits_c_u, label_p, reduction="none") * mask).sum() / mask.sum()

        loss_x = F.cross_entropy(logits_c_x, label) + F.cross_entropy(logits_x, label) + F.cross_entropy(logits_a_x, label)

        return logits_x, loss_x, loss_u



@TRAINER_REGISTRY.register()
class PMCC(BaseDA):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model_vit = load_clip_to_cpu_teacher(cfg)

        if cfg.TRAINER.PMCC.PREC == "fp32" or cfg.TRAINER.PMCC.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model, clip_model_vit)

        print("Turning off gradients in both the image and the text encoder...")
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "prompt_learner" in name:
                param.requires_grad_(True)
            if "classifier_layer" in name:
                param.requires_grad_(True)
            if "bank" in name:
                param.requires_grad_(False)

        Total_Memory = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                Total_Memory += param.numel() * param.element_size() / (1024 ** 2)
                print(str(name) + " " + str(param.requires_grad) + " " + str(
                    (param.numel() * param.element_size()) / (1024 ** 2)) + "MB")
        print("Model Total Memory : " + str(Total_Memory) + "MB")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

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
            raise ValueError('Training batch name is wrong!')

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PMCC.PREC == "amp" else None

        # self.construct_bank()

    def forward_backward(self, batch_x, batch_u):
        prec = self.cfg.TRAINER.PMCC.PREC
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)

        if prec == "amp":
            with autocast():
                output_x, loss_x, loss_u = self.model(image_x, label, image_u, epoch=self.epoch, train=True)

                loss = loss_x + loss_u

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_x, loss_x, loss_u = self.model(image_x, label, image_u, epoch=self.epoch, train=True)

            loss = loss_x + loss_u
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "acc_x": compute_accuracy(output_x, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        input = batch_x["img"]
        label = batch_x["label"]
        input_u = batch_u["img"]

        input = input.to(self.device)
        label = label.to(self.device)
        input_u = input_u.to(self.device)
        return input, label, input_u

    @torch.no_grad()
    def construct_bank(self):
        self.set_model_mode("eval")
        print("Constructing source feature bank...")
        data_loader_x = self.train_loader_x
        for batch_idx, batch in enumerate(data_loader_x):
            input, label = self.parse_batch_test(batch)
            self.model(input, label=label, construct=True)
            if min(self.model.source_max_probs_list) > 0.99:
                break

        print('Feature banks are completed!')

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        time_start = time.time()
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)
        elapsed = round((time.time() - time_start) * 1000)
        print(str(elapsed))

        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        results_all = results["accuracy"]

        return results_all