def _deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


EXPERIMENT_PRESETS = {
    "untuned_robustness": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "resnet18",
            "depth_backbone": "resnet18",
            "head_type": "transformer",
            "fusion_dim": 512,
            "fusion_grid_size": 7,
        },
        "dataset_overrides": {
            "augmentation": {
                "lowlight": {
                    "enable_train": False,
                    "enable_eval": True,
                }
            }
        },
        "train_kwargs": {},
    },
    "untuned_adapt": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "resnet18",
            "depth_backbone": "resnet18",
            "head_type": "transformer",
            "fusion_dim": 512,
            "fusion_grid_size": 7,
        },
        "dataset_overrides": {
            "augmentation": {
                "lowlight": {
                    "enable_train": True,
                    "enable_eval": True,
                }
            }
        },
        "train_kwargs": {},
    },
    # Best normal-light configuration so far:
    #   - structure: VV + MLP
    #   - lr: 4e-5
    #   - weight_decay: 1e-3
    #   - scheduler_patience: 3
    #   - dropout: 0.1
    #   - optimizer: AdamW
    #   - label_smoothing: 0.1
    "rr_transformer": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "resnet18",
            "depth_backbone": "resnet18",
            "head_type": "transformer",
        },
        "train_kwargs": {},
    },
    "rr_mlp": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "resnet18",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "train_kwargs": {},
    },
    "vv_transformer": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "transformer",
        },
        "train_kwargs": {},
    },
    "rv_transformer": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "resnet18",
            "depth_backbone": "vit",
            "head_type": "transformer",
        },
        "train_kwargs": {},
    },
    "vr_transformer": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "transformer",
        },
        "train_kwargs": {},
    },
    "vv_mlp": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {},
    },
    "vv_mlp_tuned": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 5e-5,
            "weight_decay": 1e-2,
            "fusion_dim": 768,
            "fusion_grid_size": 14,
            "scheduler_patience": 4,
        },
    },
    "vv_mlp_lr7e5": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 7e-5,
        },
    },
    "vv_mlp_wd1e3": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "weight_decay": 1e-3,
        },
    },
    "vv_mlp_lr7e5_wd1e3": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 7e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
        },
    },
    "vv_mlp_lr5e5_wd1e3": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 5e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
        },
    },
    "vv_mlp_best_current": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 4e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
        },
    },
    "normal_best_final": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 4e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
        },
    },
    "vv_mlp_lr6e5_wd1e3": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 6e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
        },
    },
    "vv_mlp_lr4e5_wd1e3": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 4e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
        },
    },
    "vv_mlp_best_current_dropout0": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 4e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
            "dropout": 0.0,
        },
    },
    "vv_mlp_best_current_dropout02": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 4e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
            "dropout": 0.2,
        },
    },
    "vv_mlp_best_current_ls005": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 4e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
            "label_smoothing": 0.05,
        },
    },
    "vv_mlp_best_current_ls0": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 4e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
            "label_smoothing": 0.0,
        },
    },
    "vv_mlp_best_current_adam": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 4e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
            "optimizer_name": "adam",
        },
    },
    "vv_mlp_best_current_adapt": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "dataset_overrides": {
            "augmentation": {
                "lowlight": {
                    "enable_train": True,
                    "enable_eval": True,
                }
            }
        },
        "train_kwargs": {
            "lr": 4e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
        },
    },
    "rv_mlp": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "resnet18",
            "depth_backbone": "vit",
            "head_type": "mlp",
        },
        "train_kwargs": {},
    },
    "vr_mlp": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "train_kwargs": {},
    },
    "vr_mlp_tuned": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 5e-5,
            "weight_decay": 1e-2,
            "fusion_dim": 768,
            "fusion_grid_size": 14,
            "scheduler_patience": 4,
        },
    },
    "vr_mlp_lr7e5": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 7e-5,
        },
    },
    "vr_mlp_wd1e3": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "weight_decay": 1e-3,
        },
    },
    "vr_mlp_lr7e5_wd1e3": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 7e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
        },
    },
    "vr_mlp_lr5e5_wd1e3": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 5e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 3,
        },
    },
    "vr_mlp_best_current": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "scheduler_patience": 2,
        },
    },
    # Best low-light adaptation configuration so far:
    #   - structure: VR + MLP
    #   - lowlight_train: True
    #   - lowlight_eval: True
    #   - lr: 8e-5
    #   - weight_decay: 1e-3
    #   - scheduler_patience: 2
    "vr_mlp_best_current_adapt": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "dataset_overrides": {
            "augmentation": {
                "lowlight": {
                    "enable_train": True,
                    "enable_eval": True,
                }
            }
        },
        "train_kwargs": {
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "scheduler_patience": 2,
        },
    },
    "lowlight_best_final": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "dataset_overrides": {
            "augmentation": {
                "lowlight": {
                    "enable_train": True,
                    "enable_eval": True,
                }
            }
        },
        "train_kwargs": {
            "lr": 8e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 2,
        },
    },
    "vr_mlp_lr1e4_wd2e3": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 1e-4,
            "weight_decay": 2e-3,
            "scheduler_patience": 2,
        },
    },
    "vr_mlp_lr1e4_wd2e3_adapt": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "dataset_overrides": {
            "augmentation": {
                "lowlight": {
                    "enable_train": True,
                    "enable_eval": True,
                }
            }
        },
        "train_kwargs": {
            "lr": 1e-4,
            "weight_decay": 2e-3,
            "scheduler_patience": 2,
        },
    },
    "vr_mlp_lr8e5_wd1e3": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "train_kwargs": {
            "lr": 8e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 2,
        },
    },
    "vr_mlp_lr8e5_wd1e3_adapt": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "dataset_overrides": {
            "augmentation": {
                "lowlight": {
                    "enable_train": True,
                    "enable_eval": True,
                }
            }
        },
        "train_kwargs": {
            "lr": 8e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 2,
        },
    },
    "vr_mlp_lr7e5_wd1e3_adapt": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "dataset_overrides": {
            "augmentation": {
                "lowlight": {
                    "enable_train": True,
                    "enable_eval": True,
                }
            }
        },
        "train_kwargs": {
            "lr": 7e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 2,
        },
    },
    "vr_mlp_lr9e5_wd1e3_adapt": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "resnet18",
            "head_type": "mlp",
        },
        "dataset_overrides": {
            "augmentation": {
                "lowlight": {
                    "enable_train": True,
                    "enable_eval": True,
                }
            }
        },
        "train_kwargs": {
            "lr": 9e-5,
            "weight_decay": 1e-3,
            "scheduler_patience": 2,
        },
    },
    "vv_transformer_tuned": {
        "model_cfg": {
            "name": "resnet18vit_midfusion",
            "rgb_backbone": "vit",
            "depth_backbone": "vit",
            "head_type": "transformer",
        },
        "train_kwargs": {
            "lr": 5e-5,
            "weight_decay": 1e-2,
            "fusion_dim": 768,
            "fusion_grid_size": 14,
        },
    },
}


def apply_experiment_preset(cfg, preset_name: str):
    if preset_name not in EXPERIMENT_PRESETS:
        available = ", ".join(sorted(EXPERIMENT_PRESETS))
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")

    preset = EXPERIMENT_PRESETS[preset_name]
    model_cfg = cfg["dataset"]["model"]
    model_cfg.update(preset["model_cfg"])
    if "dataset_overrides" in preset:
        _deep_update(cfg["dataset"], preset["dataset_overrides"])
    return dict(preset["train_kwargs"])
