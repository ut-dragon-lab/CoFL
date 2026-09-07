"""Optional pretrained SigLIP backbone; transformers is imported only at construction."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

import torch
from torch import nn


def processor_from_files(files: dict[str, bytes]):
    """Load a CPU SigLIP processor from embedded files, without model weights.

    DataLoader workers can reconstruct only their image processor and tokenizer.
    All assets are local, and temporary files are removed before returning.
    Loaded tokenizers retain their vocabulary in memory.
    """
    if (
        not isinstance(files, dict)
        or not {"preprocessor_config.json", "tokenizer_config.json"} <= files.keys()
    ):
        raise ValueError("SigLIP assets require complete processor files")
    for name, content in files.items():
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or ":" in name
            or PurePosixPath(name).is_absolute()
            or any(part in {".", ".."} for part in name.split("/"))
            or name != PurePosixPath(name).as_posix()
            or type(content) is not bytes
        ):
            raise ValueError("Processor files require relative paths and bytes")

    from transformers import SiglipProcessor

    with TemporaryDirectory(prefix="cofl-siglip-processor-") as directory:
        root = Path(directory)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        return SiglipProcessor.from_pretrained(
            root, local_files_only=True, use_fast=True, trust_remote_code=False
        )


class SigLIPBackbone(nn.Module):
    """SigLIP vision/text modules with pooled text prepended to token features. Use local_files_only=True to require cached assets."""

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        unfreeze_last_n_layers: int = 0,
        max_text_length: int = 64,
        deterministic_embeddings: bool = True,
        local_files_only: bool = False,
        vision_unfreeze_last_n_layers: int | None = None,
        text_unfreeze_last_n_layers: int | None = None,
    ):
        super().__init__()
        self._resolve_freeze_strategy(
            unfreeze_last_n_layers,
            vision_unfreeze_last_n_layers,
            text_unfreeze_last_n_layers,
        )
        try:
            from transformers import SiglipProcessor, SiglipTextModel, SiglipVisionModel
        except ImportError as exc:
            raise ImportError(
                "Vision-language policies require the vision extra: pip install 'cofl-navigation[vision]'"
            ) from exc
        self.local_files_only = bool(local_files_only)
        self.model_name = model_name
        self.max_text_length = int(max_text_length)
        self.deterministic_embeddings = bool(deterministic_embeddings)
        self.processor = SiglipProcessor.from_pretrained(
            model_name, local_files_only=local_files_only, use_fast=True
        )
        self._processor_files: dict[str, bytes] | None = None
        self.vision_encoder = SiglipVisionModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.text_encoder = SiglipTextModel.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.set_freeze_strategy(
            unfreeze_last_n_layers=unfreeze_last_n_layers,
            vision_unfreeze_last_n_layers=vision_unfreeze_last_n_layers,
            text_unfreeze_last_n_layers=text_unfreeze_last_n_layers,
        )
        self.vision_dim = self.vision_encoder.config.hidden_size
        self.text_dim = self.text_encoder.config.hidden_size

    def export_assets(self) -> dict:
        """Export reconstruction assets for a self-contained checkpoint.

        The bundle contains model configurations and every file emitted by
        the processor, including binary tokenizer vocabularies. Model weights
        remain exclusively in the enclosing policy's ``state_dict``. Values
        are plain containers, JSON scalars and bytes, so no custom classes are
        needed by ``torch.load(weights_only=True)``.

        Processor assets are fixed at the first export and retained as bytes.
        This also permits re-exporting a restored SentencePiece tokenizer after
        its temporary vocabulary file has been removed.
        """
        if self._processor_files is None:
            with TemporaryDirectory(prefix="cofl-siglip-export-") as directory:
                root = Path(directory)
                self.processor.save_pretrained(root)
                self._processor_files = {
                    path.relative_to(root).as_posix(): path.read_bytes()
                    for path in sorted(root.rglob("*"))
                    if path.is_file()
                }

        def model_config(model):
            config = model.config.to_dict()
            # The checkpoint carries weights; reconstruction has no source path.
            config.pop("_name_or_path", None)
            return json.loads(json.dumps(config, allow_nan=False))

        return {
            "vision_config": model_config(self.vision_encoder),
            "text_config": model_config(self.text_encoder),
            "processor_files": dict(self._processor_files),
            "max_text_length": self.max_text_length,
            "unfreeze_last_n_layers": self.unfreeze_last_n_layers,
            "vision_unfreeze_last_n_layers": self.vision_unfreeze_last_n_layers,
            "text_unfreeze_last_n_layers": self.text_unfreeze_last_n_layers,
            "deterministic_embeddings": self.deterministic_embeddings,
        }

    @classmethod
    def from_assets(cls, assets: dict) -> SigLIPBackbone:
        """Construct from embedded assets using transformers 4.57.1.

        No pretrained model weights, Hub access or external cache are used.
        The caller must restore the enclosing policy's ``state_dict`` before
        inference or training. Processor files exist only while loading;
        tokenization and subsequent exports work after that directory closes.
        """
        required = {
            "vision_config",
            "text_config",
            "processor_files",
            "max_text_length",
            "unfreeze_last_n_layers",
            "deterministic_embeddings",
        }
        optional = {"vision_unfreeze_last_n_layers", "text_unfreeze_last_n_layers"}
        if (
            not isinstance(assets, dict)
            or not required <= set(assets)
            or not set(assets) <= required | optional
        ):
            raise ValueError("SigLIP assets must contain the complete reconstruction bundle")
        max_length = assets["max_text_length"]
        unfreeze = assets["unfreeze_last_n_layers"]
        vision_unfreeze = assets.get("vision_unfreeze_last_n_layers")
        text_unfreeze = assets.get("text_unfreeze_last_n_layers")
        deterministic = assets["deterministic_embeddings"]
        if type(max_length) is not int or max_length < 1:
            raise ValueError("SigLIP max_text_length must be a positive integer")
        cls._resolve_freeze_strategy(unfreeze, vision_unfreeze, text_unfreeze)
        if type(deterministic) is not bool:
            raise ValueError("SigLIP deterministic_embeddings must be a boolean")
        for name, model_type in (
            ("vision_config", "siglip_vision_model"),
            ("text_config", "siglip_text_model"),
        ):
            config = assets[name]
            if not isinstance(config, dict) or config.get("model_type") != model_type:
                raise ValueError("Invalid SigLIP " + name)
        from transformers import (
            SiglipTextConfig,
            SiglipTextModel,
            SiglipVisionConfig,
            SiglipVisionModel,
        )

        vision_config = SiglipVisionConfig.from_dict(assets["vision_config"])
        text_config = SiglipTextConfig.from_dict(assets["text_config"])
        if max_length > text_config.max_position_embeddings:
            raise ValueError("max_text_length exceeds the embedded text configuration")
        backbone = cls.__new__(cls)
        nn.Module.__init__(backbone)
        backbone.local_files_only = True
        backbone.model_name = "checkpoint"
        backbone.max_text_length = max_length
        backbone.deterministic_embeddings = deterministic
        backbone.processor = processor_from_files(assets["processor_files"])
        backbone._processor_files = dict(assets["processor_files"])
        backbone.vision_encoder = SiglipVisionModel(vision_config)
        backbone.text_encoder = SiglipTextModel(text_config)
        backbone.set_freeze_strategy(
            unfreeze_last_n_layers=unfreeze,
            vision_unfreeze_last_n_layers=vision_unfreeze,
            text_unfreeze_last_n_layers=text_unfreeze,
        )
        backbone.vision_dim = vision_config.hidden_size
        backbone.text_dim = text_config.hidden_size
        return backbone

    @staticmethod
    def _resolve_freeze_strategy(
        unfreeze_last_n_layers: int,
        vision_unfreeze_last_n_layers: int | None,
        text_unfreeze_last_n_layers: int | None,
    ) -> tuple[int, int]:
        for name, count in (
            ("unfreeze_last_n_layers", unfreeze_last_n_layers),
            ("vision_unfreeze_last_n_layers", vision_unfreeze_last_n_layers),
            ("text_unfreeze_last_n_layers", text_unfreeze_last_n_layers),
        ):
            if count is None and name != "unfreeze_last_n_layers":
                continue
            if type(count) is not int or count < -1:
                raise ValueError(f"{name} must be -1 or a nonnegative integer")
        return (
            unfreeze_last_n_layers
            if vision_unfreeze_last_n_layers is None
            else vision_unfreeze_last_n_layers,
            unfreeze_last_n_layers
            if text_unfreeze_last_n_layers is None
            else text_unfreeze_last_n_layers,
        )

    def set_freeze_strategy(
        self,
        *,
        unfreeze_last_n_layers: int = 0,
        vision_unfreeze_last_n_layers: int | None = None,
        text_unfreeze_last_n_layers: int | None = None,
    ) -> None:
        """Set tower-specific training flags, including for restored backbones.

        Each tower uses its explicit value or falls back to the shared value:
        0 freezes it, -1 trains its entire module, and N trains its last N
        transformer blocks. Partial unfreezing leaves embeddings, final norms,
        and pooling heads frozen. Both depths are checked before flags change.
        Apply this before constructing an optimizer; it does not rebuild an
        existing optimizer's parameter groups or change train/eval mode.
        """
        vision_count, text_count = self._resolve_freeze_strategy(
            unfreeze_last_n_layers,
            vision_unfreeze_last_n_layers,
            text_unfreeze_last_n_layers,
        )
        vision_layers = self.vision_encoder.vision_model.encoder.layers
        text_layers = self.text_encoder.text_model.encoder.layers
        for name, count, layers in (
            ("vision", vision_count, vision_layers),
            ("text", text_count, text_layers),
        ):
            if count > len(layers):
                raise ValueError(f"{name}_unfreeze_last_n_layers exceeds the {name} encoder depth")
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.vision_unfreeze_last_n_layers = vision_count
        self.text_unfreeze_last_n_layers = text_count
        for encoder, layers, count in (
            (self.vision_encoder, vision_layers, vision_count),
            (self.text_encoder, text_layers, text_count),
        ):
            encoder.requires_grad_(count == -1)
            if count > 0:
                for layer in layers[-count:]:
                    layer.requires_grad_(True)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def patch_size(self):
        return int(self.vision_encoder.config.patch_size)

    @contextlib.contextmanager
    def _deterministic_eval_context(self):
        if not self.deterministic_embeddings:
            yield
            return
        vision_was_training = bool(self.vision_encoder.training)
        text_was_training = bool(self.text_encoder.training)
        try:
            self.vision_encoder.eval()
            self.text_encoder.eval()
            yield
        finally:
            self.vision_encoder.train(vision_was_training)
            self.text_encoder.train(text_was_training)

    def preprocess(
        self,
        *,
        images: torch.Tensor | Any | Sequence[Any] | None = None,
        texts: str | Sequence[str] | None = None,
        padding: str = "max_length",
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str = "pt",
    ) -> dict:
        """Return processed inputs and an explicit tokenizer mask for text fusion.

        The mask includes real EOS tokens. It is not inferred from padding IDs,
        which some tokenizers share with EOS. Tensor encoder inputs must already
        be preprocessed.
        """
        max_length = int(self.max_text_length if max_length is None else max_length)
        kwargs = {"return_tensors": return_tensors}
        if texts is not None:
            kwargs.update(
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_attention_mask=True,
            )
        return self.processor(images=images, text=texts, **kwargs)

    def encode_image(
        self, images_or_pixel_values: torch.Tensor | Any | Sequence[Any]
    ) -> torch.Tensor:
        if isinstance(images_or_pixel_values, torch.Tensor):
            pixel_values_t = images_or_pixel_values
        else:
            processed = self.preprocess(images=images_or_pixel_values)
            pixel_values_t = processed["pixel_values"]
        pixel_values_t = pixel_values_t.to(self.device)
        with (
            self._deterministic_eval_context(),
            torch.set_grad_enabled(torch.is_grad_enabled() and self.training),
        ):
            outputs = self.vision_encoder(pixel_values=pixel_values_t)
        return outputs.last_hidden_state

    def encode_text(
        self,
        text_or_input_ids: torch.Tensor | str | Sequence[str],
        attention_mask: torch.Tensor | None = None,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Return [B,L+1,D] pooled/token features with SigLIP's unmasked default encoding.

        An explicitly supplied attention_mask controls the backbone's internal
        attention. High-level policies instead use the tokenizer mask only in
        their visual-language fusion, preserving pretrained padding and pooling.
        """
        if isinstance(text_or_input_ids, torch.Tensor):
            input_ids_t = text_or_input_ids
            attention_mask_t = attention_mask
        else:
            processed = self.preprocess(texts=text_or_input_ids, max_length=max_length)
            input_ids_t = processed["input_ids"]
            attention_mask_t = attention_mask
        input_ids_t = input_ids_t.to(self.device)
        if attention_mask_t is not None:
            attention_mask_t = attention_mask_t.to(self.device)
        with (
            self._deterministic_eval_context(),
            torch.set_grad_enabled(torch.is_grad_enabled() and self.training),
        ):
            outputs = self.text_encoder(input_ids=input_ids_t, attention_mask=attention_mask_t)
        return torch.cat([outputs.pooler_output.unsqueeze(1), outputs.last_hidden_state], dim=1)

    def forward(
        self,
        images_or_pixel_values: torch.Tensor | Any | Sequence[Any],
        text_or_input_ids: torch.Tensor | str | Sequence[str],
        attention_mask: torch.Tensor | None = None,
    ):
        image_features = self.encode_image(images_or_pixel_values)
        text_features = self.encode_text(text_or_input_ids, attention_mask=attention_mask)
        return (image_features, text_features)
