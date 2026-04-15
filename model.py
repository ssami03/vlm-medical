"""
model.py — Vision-Language Model with BioGPT Decoder
EEE3094 Dissertation — Sami Zarroug (220672267)

Architecture:
    CLIP ViT-B/32 (pre-trained, top CLIP blocks partially unfrozen)
        → CLS token (512-dim) → MLP classifier (13 classes)
        → 49 patch tokens (768-dim) → Bridge MLP (768→1024)
            → Prepended as visual prefix to BioGPT input
            → BioGPT self-attention over [visual_tokens | text_tokens]
            → Generated radiology report
"""

import torch
import torch.nn as nn
import clip
from transformers import BioGptForCausalLM, BioGptTokenizer

from config import (
    CLIP_MODEL_NAME,
    CLIP_EMBED_DIM,
    CLIP_HIDDEN_DIM,
    CLIP_NUM_PATCHES,
    DECODER_MODEL_NAME,
    DECODER_HIDDEN_DIM,
    NUM_CLASSES,
    CLASSIFIER_CFG,
    DECODER_CFG,
)


class MedicalVLM(nn.Module):
    """
    Full Vision-Language Model for medical chest X-ray analysis.

    Components:
        1. CLIP ViT-B/32 visual encoder
        2. Classification head (512 -> 13 classes)
        3. Bridge MLP (768 -> 1024)
        4. BioGPT decoder

    Visual Prefix Injection:
        Visual patch embeddings are projected into BioGPT embedding space
        and prepended to text token embeddings.
    """

    def __init__(self, device: torch.device, freeze_encoder: bool = True):
        super().__init__()
        self.device = device
        self._patch_features = None

        # 1. CLIP visual encoder
        clip_model, _ = clip.load(CLIP_MODEL_NAME, device="cpu")
        self.visual = clip_model.visual

        for param in self.visual.parameters():
            param.requires_grad_(False)

        if not freeze_encoder:
            n_blocks = CLASSIFIER_CFG.get("unfreeze_blocks", 2)
            for block in self.visual.transformer.resblocks[-n_blocks:]:
                for param in block.parameters():
                    param.requires_grad_(True)

        # Hook for patch tokens from last transformer block
        self._hook = self.visual.transformer.resblocks[-1].register_forward_hook(
            self._patch_hook
        )

        # 2. Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(CLIP_EMBED_DIM),
            nn.Linear(CLIP_EMBED_DIM, CLIP_EMBED_DIM),
            nn.GELU(),
            nn.Dropout(CLASSIFIER_CFG["dropout"]),
            nn.Linear(CLIP_EMBED_DIM, NUM_CLASSES),
        )

        # 3. Bridge MLP
        self.bridge = nn.Sequential(
            nn.Linear(CLIP_HIDDEN_DIM, DECODER_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(DECODER_HIDDEN_DIM, DECODER_HIDDEN_DIM),
            nn.LayerNorm(DECODER_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 4. BioGPT decoder
        self.decoder = BioGptForCausalLM.from_pretrained(DECODER_MODEL_NAME)
        self.decoder.config.tie_word_embeddings = False

        self._print_summary()

    def _print_summary(self):
        enc_total = sum(p.numel() for p in self.visual.parameters())
        enc_train = sum(p.numel() for p in self.visual.parameters() if p.requires_grad)
        cls_params = sum(p.numel() for p in self.classifier.parameters())
        bridge_params = sum(p.numel() for p in self.bridge.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total = enc_total + cls_params + bridge_params + dec_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n[VLM] Architecture Summary:")
        print(f"  CLIP ViT-B/32:     {enc_total:>12,} params ({enc_train:,} trainable)")
        print(f"  Classifier head:   {cls_params:>12,} params")
        print(f"  Bridge MLP:        {bridge_params:>12,} params")
        print(f"  BioGPT decoder:    {dec_params:>12,} params (pre-trained on PubMed)")
        print(f"  Total:             {total:>12,} params ({trainable:,} trainable)")

    # ──────────────────────────────────────────────────────────────
    # Patch token hook
    # ──────────────────────────────────────────────────────────────

    def _patch_hook(self, module, input, output):
        """
        Capture 49 patch tokens from the last CLIP ViT transformer block.
        CLIP ViT output shape here is typically (seq_len=50, batch, dim=768).
        Position 0 = CLS token, positions 1: = 49 patch tokens.
        """
        patch_features = output[1:, :, :].permute(1, 0, 2).contiguous()  # (B, 49, 768)
        self._patch_features = patch_features

        # Needed for manual gradient-based patch maps and debugging
        if self._patch_features.requires_grad:
            self._patch_features.retain_grad()

    # ──────────────────────────────────────────────────────────────
    # Encoding methods
    # ──────────────────────────────────────────────────────────────

    def encode_image_cls(self, images: torch.Tensor) -> torch.Tensor:
        """CLS embedding for classification. Shape: (B, 512)"""
        return self.visual(images.float())

    def encode_image_patches(
        self, images: torch.Tensor, enable_grads: bool = False
    ) -> torch.Tensor:
        """
        Patch tokens through bridge for decoder input. Shape: (B, 49, 1024).
        """
        if enable_grads:
            _ = self.visual(images.float())
        else:
            with torch.no_grad():
                _ = self.visual(images.float())

        return self.bridge(self._patch_features)

    # ──────────────────────────────────────────────────────────────
    # Classification
    # ──────────────────────────────────────────────────────────────

    def classify(self, images: torch.Tensor) -> torch.Tensor:
        """Return raw logits (B, NUM_CLASSES)."""
        cls_features = self.encode_image_cls(images)
        return self.classifier(cls_features.float())

    # ──────────────────────────────────────────────────────────────
    # Report generation training
    # ──────────────────────────────────────────────────────────────

    def forward_decoder(self, images, input_ids, attention_mask, labels=None):
        """
        Forward pass for report generation training using visual prefix injection.
        """
        batch_size = images.shape[0]

        # Visual prefix
        visual_embeds = self.encode_image_patches(images)  # (B, 49, 1024)

        # Text embeddings
        text_embeds = self.decoder.biogpt.embed_tokens(input_ids)  # (B, T, 1024)

        # Concatenate: [visual | text]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, 49+T, 1024)

        # Attention mask
        visual_mask = torch.ones(
            batch_size,
            CLIP_NUM_PATCHES,
            dtype=attention_mask.dtype,
            device=self.device,
        )
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        # Labels
        if labels is not None:
            visual_labels = torch.full(
                (batch_size, CLIP_NUM_PATCHES),
                -100,
                dtype=labels.dtype,
                device=self.device,
            )
            combined_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            combined_labels = None

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

    # ──────────────────────────────────────────────────────────────
    # Report generation inference
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_report(
        self,
        image: torch.Tensor,
        tokenizer: BioGptTokenizer,
        max_length: int = None,
        prefix: str = "Findings:",
    ) -> str:
        """
        Greedy autoregressive decoding.
        """
        self.eval()
        if max_length is None:
            max_length = DECODER_CFG["gen_max_length"]

        if image.dim() == 3:
            image = image.unsqueeze(0)

        visual_embeds = self.encode_image_patches(image)
        prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(self.device)
        generated_ids = prefix_ids.clone()

        for _ in range(max_length):
            text_embeds = self.decoder.biogpt.embed_tokens(generated_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)

            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            next_logits = outputs.logits[0, -1, :]

            for token_id in generated_ids[0]:
                next_logits[token_id] /= 1.2

            next_token = torch.argmax(next_logits, dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat(
                [generated_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1
            )

        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_report_beam(
        self,
        image: torch.Tensor,
        tokenizer: BioGptTokenizer,
        max_new_tokens: int = None,
        prefix: str = "Findings:",
        num_beams: int = 4,
    ) -> str:
        """
        Beam search generation using HuggingFace .generate() with inputs_embeds.
        """
        self.eval()
        if max_new_tokens is None:
            max_new_tokens = DECODER_CFG["gen_max_length"]

        if image.dim() == 3:
            image = image.unsqueeze(0)

        visual_embeds = self.encode_image_patches(image)
        prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(self.device)
        text_embeds = self.decoder.biogpt.embed_tokens(prefix_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)

        try:
            output = self.decoder.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                length_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            return tokenizer.decode(output[0], skip_special_tokens=True)
        except Exception as e:
            print(f"[VLM] Beam search failed ({e}), falling back to greedy")
            return self.generate_report(image, tokenizer, max_new_tokens, prefix)

    @torch.no_grad()
    def generate_report_sampling(
        self,
        image: torch.Tensor,
        tokenizer: BioGptTokenizer,
        max_new_tokens: int = None,
        prefix: str = "Findings:",
    ) -> str:
        """
        Sampling-based generation to reduce repetitive safe-template outputs.
        """
        self.eval()
        if max_new_tokens is None:
            max_new_tokens = DECODER_CFG["gen_max_length"]

        if image.dim() == 3:
            image = image.unsqueeze(0)

        visual_embeds = self.encode_image_patches(image)
        prefix_ids = tokenizer.encode(prefix, return_tensors="pt").to(self.device)
        text_embeds = self.decoder.biogpt.embed_tokens(prefix_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)

        output = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    # ──────────────────────────────────────────────────────────────
    # Phase control methods
    # ──────────────────────────────────────────────────────────────

    def freeze_for_classification(self):
        """Phase 1: classifier head + top N visual blocks trainable."""
        for p in self.parameters():
            p.requires_grad_(False)

        for p in self.classifier.parameters():
            p.requires_grad_(True)

        n_blocks = CLASSIFIER_CFG.get("unfreeze_blocks", 2)
        for block in self.visual.transformer.resblocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VLM] Classification mode: {trainable:,} trainable params")

    def freeze_for_decoder_prewarm(self):
        """Phase 2a: only bridge MLP trainable."""
        for p in self.parameters():
            p.requires_grad_(False)

        for p in self.bridge.parameters():
            p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VLM] Prewarm mode: {trainable:,} trainable params (bridge only)")

    def freeze_for_decoder_full(self):
        """
        Phase 2b: bridge + top 2 visual blocks + BioGPT decoder trainable.
        """
        for p in self.parameters():
            p.requires_grad_(False)

        for p in self.bridge.parameters():
            p.requires_grad_(True)

        for block in self.visual.transformer.resblocks[-2:]:
            for p in block.parameters():
                p.requires_grad_(True)

        for p in self.decoder.parameters():
            p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VLM] Full decoder mode: {trainable:,} trainable params")

    def get_classifier_param_groups(self):
        """Differential learning rates: slow backbone, faster head."""
        n_blocks = CLASSIFIER_CFG.get("unfreeze_blocks", 2)
        backbone_params = []
        for block in self.visual.transformer.resblocks[-n_blocks:]:
            backbone_params.extend(list(block.parameters()))

        return [
            {"params": backbone_params, "lr": CLASSIFIER_CFG["lr_backbone"]},
            {"params": list(self.classifier.parameters()), "lr": CLASSIFIER_CFG["lr_head"]},
        ]


class CLIPClassifierForGradCAM(nn.Module):
    """
    Wrapper so pytorch-grad-cam can target classifier logits cleanly.
    """
    def __init__(self, medical_vlm: MedicalVLM):
        super().__init__()
        self.medical_vlm = medical_vlm

    def forward(self, x):
        return self.medical_vlm.classify(x)