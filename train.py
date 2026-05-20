import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from utils import get_column_normalizer, get_img_preprocessor, ModelObjectCallBack
from lewm_training.augmentations import build_image_augmentation


def _batch_size(batch):
    pixels = batch.get("pixels")
    if pixels is None or not hasattr(pixels, "shape"):
        return None
    return int(pixels.shape[0])


def _stage_prefix(stage):
    if stage in {"val", "validate", "validation"}:
        return "val"
    if stage in {"fit", "train", "training"}:
        return "train"
    return str(stage)


def _current_lr(module):
    trainer = getattr(module, "trainer", None)
    optimizers = getattr(trainer, "optimizers", None)
    if not optimizers:
        return None
    param_groups = getattr(optimizers[0], "param_groups", None)
    if not param_groups:
        return None
    lr = param_groups[0].get("lr")
    if lr is None:
        return None
    return torch.as_tensor(lr, device=module.device)


def _extra_metrics(batch, emb, ctx_emb, tgt_emb, pred_emb, act_emb, pred_loss):
    pred_error = pred_emb.detach() - tgt_emb.detach()
    pred_flat = pred_emb.detach().reshape(-1, pred_emb.shape[-1])
    tgt_flat = tgt_emb.detach().reshape(-1, tgt_emb.shape[-1])

    metrics = {
        "pred_rmse": pred_loss.detach().sqrt(),
        "pred_abs_error": pred_error.abs().mean(),
        "pred_target_cosine": torch.nn.functional.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean(),
        "embedding_norm": emb.detach().norm(dim=-1).mean(),
        "context_embedding_norm": ctx_emb.detach().norm(dim=-1).mean(),
        "target_embedding_norm": tgt_emb.detach().norm(dim=-1).mean(),
        "pred_embedding_norm": pred_emb.detach().norm(dim=-1).mean(),
        "action_embedding_norm": act_emb.detach().norm(dim=-1).mean(),
    }

    pixels = batch.get("pixels")
    if pixels is not None and hasattr(pixels, "float"):
        pixels = pixels.detach().float()
        metrics["pixels_mean"] = pixels.mean()
        metrics["pixels_std"] = pixels.std()

    return metrics


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["weighted_sigreg_loss"] = lambd * output["sigreg_loss"]
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    losses = {k: v.detach() for k, v in output.items() if "loss" in k}
    prefix = _stage_prefix(stage)
    metrics = {f"{prefix}/{k}": v for k, v in losses.items()}
    metrics.update(
        {
            f"{prefix}/{k}": v
            for k, v in _extra_metrics(batch, emb, ctx_emb, tgt_emb, pred_emb, act_emb, output["pred_loss"]).items()
        }
    )
    lr = _current_lr(self)
    if lr is not None and prefix == "train":
        metrics["train/lr"] = lr

    batch_size = _batch_size(batch)
    log_kwargs = {"logger": True, "sync_dist": True}
    if batch_size is not None:
        log_kwargs["batch_size"] = batch_size

    if prefix == "val":
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, **log_kwargs)
    else:
        self.log_dict(metrics, on_step=True, on_epoch=True, **log_kwargs)
    return output

def build_train_val_transforms(dataset, cfg):
    base_transforms = [
        get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            base_transforms.append(normalizer)

            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

    train_transforms = []
    augmentation = build_image_augmentation(cfg.get("augmentation"), source="pixels", target="pixels")
    if augmentation is not None:
        train_transforms.append(augmentation)
    train_transforms.extend(base_transforms)

    return (
        spt.data.transforms.Compose(*train_transforms),
        spt.data.transforms.Compose(*base_transforms),
    )


def build_train_val_datasets(cfg, train_transform, val_transform):
    val_dataset_cfg = cfg.data.get("val_dataset")
    if val_dataset_cfg is not None:
        train_dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=train_transform)
        merged_val_dataset_cfg = OmegaConf.merge(cfg.data.dataset, val_dataset_cfg)
        val_dataset = swm.data.HDF5Dataset(**merged_val_dataset_cfg, transform=val_transform)
        return train_dataset, val_dataset

    split_dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_idx, val_idx = spt.data.random_split(
        split_dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train_dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=train_transform)
    val_dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=val_transform)
    return (
        spt.data.Subset(train_dataset, train_idx.indices),
        spt.data.Subset(val_dataset, val_idx.indices),
    )


def _resolve_checkpoint_path(path):
    path = Path(str(path)).expanduser()
    if path.is_absolute():
        return path
    return Path(swm.data.utils.get_cache_dir(), path)


def load_initial_weights(module, cfg):
    init_ckpt_path = cfg.get("init_ckpt_path")
    if not init_ckpt_path:
        return

    path = _resolve_checkpoint_path(init_ckpt_path)
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing[:20]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:20]}")
        raise RuntimeError(f"Initial checkpoint did not match model: {'; '.join(details)}")
    print(f"[train] initialized weights from {path}")


@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    stats_dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    train_transform, val_transform = build_train_val_transforms(stats_dataset, cfg)
    train_set, val_set = build_train_val_datasets(cfg, train_transform, val_transform)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    trainer_max_steps = cfg.trainer.get("max_steps")
    if trainer_max_steps is not None and int(trainer_max_steps) > 0:
        scheduler_max_steps = int(trainer_max_steps)
    else:
        scheduler_max_steps = len(train) * int(cfg.trainer.max_epochs)
    scheduler_warmup_steps = max(1, min(1000, scheduler_max_steps // 20))
    
    ##############################
    ##       model / optim      ##
    ##############################

    encoder = spt.backbone.utils.vit_hf(
        cfg.encoder_scale,
        patch_size=cfg.patch_size,
        image_size=cfg.img_size,
        pretrained=False,
        use_mask_token=False,
    )

    hidden_dim = encoder.config.hidden_size
    embed_dim = cfg.wm.get("embed_dim", hidden_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)
    
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": scheduler_warmup_steps,
                "max_steps": scheduler_max_steps,
            },
            "interval": "step",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )
    load_initial_weights(world_model, cfg)

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
