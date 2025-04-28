import os
import random
import sys
from itertools import chain

import numpy
import pandas as pd
from matplotlib import pyplot as plt
from openTSNE import TSNE
from torch.cuda.amp import GradScaler, autocast

from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights
from dassl.optim import build_optimizer, build_lr_scheduler

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.baseda import *

from utils.clip_part import *
from utils.quaternion_layers import QuaternionLinearAutograd
from utils.templates import CUSTOM_TEMPLATES
from utils.single_ift_block import *


_tokenizer = _Tokenizer()


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

        self.vctx = None
        if self.vp:
            vctx_vectors = torch.empty(n_ctx, self.hidden_size, dtype=dtype)
            nn.init.normal_(vctx_vectors, std=0.02)
            self.vctx = nn.Parameter(vctx_vectors)

        self.deep_ctx = None
        if self.t_deep:
            if self.deep_layer == None:
                deep_ctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, ctx_dim)
            else:
                deep_ctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] + 1, self.num_tokens, ctx_dim)
            nn.init.normal_(deep_ctx_vectors, std=0.02)
            self.deep_ctx = nn.Parameter(deep_ctx_vectors)

        self.deep_vctx = None
        if self.v_deep and not self.deep_share:
            if self.deep_layer == None:
                deep_vctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, self.hidden_size)
            elif self.deep_layer != None:
                deep_vctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] - 1, self.num_tokens, self.hidden_size)
            nn.init.normal_(deep_vctx_vectors, std=0.02)
            self.deep_vctx = nn.Parameter(deep_vctx_vectors)
        elif self.v_deep and self.deep_share:
            single_layer = QuaternionLinearAutograd(ctx_dim, 768)
            # single_layer = nn.Linear(ctx_dim, self.hidden_size)
            if self.share_layer == None and self.deep_layer == None:
                deep_vctx_vectors = torch.empty(self.num_layer - 1, self.num_tokens, self.hidden_size)
                self.deep_prompt_proj = get_clones(single_layer, self.num_layer - 1)
            elif self.share_layer != None and self.deep_layer == None:
                deep_vctx_vectors = torch.empty(self.num_layer - self.share_layer[1] - 1, self.num_tokens, self.hidden_size)
                self.deep_prompt_proj = get_clones(single_layer, self.share_layer[1] - self.share_layer[0] + 1)
            elif self.share_layer != None and self.deep_layer != None:
                deep_vctx_vectors = torch.empty(self.deep_layer[1] - self.share_layer[1], self.num_tokens, self.hidden_size)
                self.deep_prompt_proj = get_clones(single_layer, self.share_layer[1] - self.share_layer[0] + 1)
            else:
                raise ValueError('deep layer and share layer are not compatible!')
            nn.init.normal_(deep_vctx_vectors, std=0.02)
            self.deep_vctx = nn.Parameter(deep_vctx_vectors)

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
        self.device = torch.device("cuda:{}".format(cfg.GPU))

        domains = cfg.DOMAINS
        if cfg.DATASET.NAME == "OfficeHome":
            DOMAINS = {'a': "art", 'c': "clipart", 'p': "product", 'r': "real_world"}
        elif cfg.DATASET.NAME == "VisDA17":
            DOMAINS = {'s': "synthetic", 'r': "real"}
        elif cfg.DATASET.NAME == "Office31":
            DOMAINS = {'a': "amazon", 'w': "webcam", 'd': "dslr"}
        elif cfg.DATASET.NAME == "DomainNet":
            DOMAINS = {'c': "clipart", 'i': "infograph", 'p': "painting", 'q': "quickdraw", 'r': "real", 's': "sketch"}

        source_domain, target_domain = domains.split('-')[0], domains.split('-')[1]

        self.K = 5
        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.source_feat_bank = torch.load("/data/dxw/PMCC/assets/" + cfg.DATASET.NAME + "/" + DOMAINS[source_domain] + "Centroid.pt", map_location=self.device)
        else:  # RN50, RN101
            self.source_feat_bank = torch.load("/data/dxw/PMCC/assets/" + cfg.DATASET.NAME + "/" + DOMAINS[source_domain] + "Centroid_RN.pt", map_location=self.device)

    def forward(self):
        vctx = self.vctx
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)   # [65, 16, 512]

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        if self.deep_share:
            deep_vctx = []
            for index, layer in enumerate(self.deep_prompt_proj):
                deep_vctx.append(layer(self.deep_ctx[index]))
            deep_vctx = torch.stack(deep_vctx)
            deep_vctx = torch.cat((deep_vctx, self.deep_vctx), dim=0)
        else:
            deep_vctx = self.deep_vctx

        return prompts, self.deep_ctx, vctx, deep_vctx

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
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.K = self.prompt_learner.K
        self.dim = clip_model.text_projection.shape[1]
        self.n_cls = len(classnames)
        self.device = torch.device("cuda:{}".format(cfg.GPU))

        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)

        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model)
            self.text_features_u = torch.load(
                "/data/dxw/PMCC/assets/VIT/text_features" + cfg.DATASET.NAME + ".pt", map_location=self.device).half().to(self.device)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model)
            self.text_features_u = torch.load(
                "/data/dxw/PMCC/assets/RN/text_features" + cfg.DATASET.NAME + ".pt", map_location=self.device).half().to(self.device)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        source_bank = torch.mean(self.prompt_learner.source_feat_bank.reshape(self.n_cls, self.K, self.dim), dim=1).to(self.device)
        self.bank = 0.5 * source_bank + 0.5 * self.text_features_u

        self.classifier_layer = nn.Sequential(
            nn.BatchNorm1d(self.dim),
            nn.LayerNorm(self.dim, eps=1e-6),
            nn.Linear(self.dim, self.n_cls, bias=False)).half()
        self.classifier_layer.apply(weights_init_classifier).to(self.device)

        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts_u = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts_u = clip.tokenize(prompts_u)

        self.confi = cfg.CONFI
        self.epoch = cfg.EPOCH
        self.warm_up = cfg.WARM_UP
        self.alpha = cfg.ALPHA
        self.beta = cfg.BETA



    def forward(self, image_x, label=None, image_u=None, epoch=None):

        prompts, deep_ctx, vctx, deep_vctx = self.prompt_learner()
        logit_scale = self.logit_scale.exp()

        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_ctx)        # [n_cls, 512] [n_cls, 77, 512]
        image_features_x = self.image_encoder(image_x.type(self.dtype), vctx, deep_vctx)    # [B, 512] [B, 196 + 1 + n_ctk + n_ctk, 512]

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features_x = (image_features_x / image_features_x.norm(dim=-1, keepdim=True)).half()

        logits_x = logit_scale * image_features_x @ text_features.t()
        logits_a_x = self.classifier_layer(image_features_x)
        logits_c_x, Fsx = self.prompt_learner.attn_block(text_features, image_features_x, self.bank)


        if not self.training:
            return 0.7*logits_x + 0.2*logits_a_x + 0.3*logits_c_x

        F_u = self.image_encoder(image_u.type(self.dtype), None, None)
        F_u = F_u / F_u.norm(dim=-1, keepdim=True)

        logits_clip = logit_scale * F_u @ self.text_features_u.t()

        image_features_u = self.image_encoder(image_u.type(self.dtype), vctx, deep_vctx)  # [B, 512] [B, 196 + 1 + n_ctk + n_ctk, 512]
        image_features_u = (image_features_u / image_features_u.norm(dim=-1, keepdim=True)).half()

        logits_u = logit_scale * image_features_u @ text_features.t()
        logits_a_u = self.classifier_layer(image_features_u)
        logits_c_u, Fsu = self.prompt_learner.attn_block(text_features, image_features_u, self.bank)

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
            loss_u += self.alpha * (F.cross_entropy(logits_a_u, label_p, reduction="none") * mask).sum() / mask.sum()
            loss_u += self.beta * (F.cross_entropy(logits_c_u, label_p, reduction="none") * mask).sum() / mask.sum()

        loss_x = F.cross_entropy(logits_x, label) + self.alpha * F.cross_entropy(logits_a_x, label) + self.beta * F.cross_entropy(logits_c_x, label)

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

        if cfg.TRAINER.PMCC.PREC == "fp32" or cfg.TRAINER.PMCC.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

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
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PMCC.PREC == "amp" else None


    def forward_backward(self, batch_x, batch_u):
        prec = self.cfg.TRAINER.PMCC.PREC
        image_x, label, image_u = self.parse_batch_train(batch_x, batch_u)

        if prec == "amp":
            with autocast():
                output_x, loss_x, loss_u = self.model(image_x, label, image_u, epoch=self.epoch)

                loss = loss_x + loss_u

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output_x, loss_x, loss_u = self.model(image_x, label, image_u, epoch=self.epoch)

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
    def T_SNE_combined(self):
        self.set_model_mode("eval")

        all_embeddings = []
        all_labels = []

        combined_loader = chain(self.train_loader_x, self.train_loader_u)

        for batch_idx, batch in enumerate(combined_loader):
            input, label = self.parse_batch_test(batch)
            prompts, deep_ctx, vctx, deep_vctx = self.model.prompt_learner()

            text_features = self.model.text_encoder(prompts, self.model.tokenized_prompts,
                                              deep_ctx)  # [n_cls, 512] [n_cls, 77, 512]
            image_features = self.model.image_encoder(input.type(self.model.dtype), vctx,
                                                deep_vctx)  # [B, 512] [B, 196 + 1 + n_ctk + n_ctk, 512]

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = (image_features / image_features.norm(dim=-1, keepdim=True)).half()

            image_features = self.model.prompt_learner.attn_block(text_features, image_features, self.model.bank)

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

        plt.scatter(embeddings[source_mask, 0], embeddings[source_mask, 1], color='blue', marker='o', s=8 , label='Source domain', alpha=0.8)
        plt.scatter(embeddings[target_mask, 0], embeddings[target_mask, 1], color='red', marker='o', s=8, label='Target domain', alpha=0.8)

        # 添加图例和其他装饰
        # plt.legend()
        plt.xticks(())
        plt.yticks(())

        out_dir = "/data/dxw/PMCC/tsne/PMCC/OfficeHome"

        plt.title('PMCC' + ' (' + str(self.domains).upper() + ')', fontdict={"family": "Times New Roman", "size": 64})

        plt.savefig(out_dir + '/PMCC-' + str(self.domains).upper() + '.pdf')

    def after_train(self):
        print("Finish training")

        do_test = not self.cfg.TEST.NO_TEST
        if do_test:
            if self.cfg.TEST.FINAL_MODEL == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")

            output_dir = self.cfg.OUTPUT_DIR
            path_parts = output_dir.split('/')
            results_file = '/'.join(path_parts[:10]) + '/' + self.cfg.DATASET.NAME + ".csv"
            print(results_file)

            file_exists = os.path.isfile(results_file)

            if self.cfg.DATASET.NAME == "VisDA17":
                result_all, accs = self.test()
                columns = ['best'] + ['acc_{}'.format(i + 1) for i in range(len(accs))] + ['avg']

                # 初始化DataFrame
                if not file_exists:
                    df = pd.DataFrame(columns=columns)
                else:
                    df = pd.read_csv(results_file)

                row_data = {'best': "best_val"}  # epoch从1开始计数
                for i, acc in enumerate(accs):
                    row_data['acc_{}'.format(i + 1)] = acc

                row_data['avg'] = result_all

                df = df.append(row_data, ignore_index=True)
                df.to_csv(results_file, index=False)

            else:
                result_all = self.test()
                columns = [
                    'a-c', 'a-p', 'a-r', 'c-a', 'c-p', 'c-r',
                    'p-a', 'p-c', 'p-r', 'r-c', 'r-a', 'r-p'
                ]

                if not file_exists:
                    # 初始化DataFrame并添加一行best_val
                    df = pd.DataFrame(columns=columns + ['avg'])
                    df.loc['best_val'] = [None] * len(columns + ['avg'])  # 使用'best_val'作为行索引
                else:
                    # 如果文件存在，读取时指定第一列作为索引
                    df = pd.read_csv(results_file, index_col=0)
                # 更新特定域的数据
                df.at['best_val', self.domains] = round(result_all, 2)

                if all(pd.notna(df.loc['best_val', col]) for col in columns):
                    # 计算平均值并更新到'avg'列
                    avg_value = df.loc['best_val', columns].mean()
                    df.at['best_val', 'avg'] = round(avg_value, 2)

                # 保存DataFrame到CSV，包含索引
                df.to_csv(results_file)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            self.evaluator.process(output, label)

        if self.cfg.DATASET.NAME == "VisDA17":
            results, accs = self.evaluator.evaluate()
        else:
            results = self.evaluator.evaluate()

        # results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        if self.cfg.DATASET.NAME == "VisDA17":
            results_all = results["perclass_accuracy"]
            return results_all, accs
        else:
            results_all = results["accuracy"]
            return results_all