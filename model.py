"""
model.py — Vision-Language Model with BioGPT Decoder
EEE3094 Dissertation — Sami Zarroug (220672267)

Architecture:
    CLIP ViT-B/32 (pre-trained, top 2 blocks unfrozen for classification)
        → CLS token (512-dim) → MLP classifier (13 classes)
        → 49 patch tokens (768-dim) → Bridge MLP (768→1024)
            → Prepended as visual prefix to BioGPT input
            → BioGPT self-attention over [visual_tokens | text_tokens]
            → Generated radiology report

Why BioGPT instead of distilgpt2:
    1. Pre-trained on 15M PubMed abstracts — knows medical vocabulary
    2. distilgpt2 trained on Reddit-linked web text — knows internet slang
    3. "pleural effusion" is in BioGPT's training data; distilgpt2 has never seen it

Why Visual Prefix instead of Cross-Attention:
    1. distilgpt2 cross-attention required add_cross_attention=True config hack
       → produced MISSING keys for all crossattention weights
       → randomly initialized cross-attention layers never converged properly
    2. Visual prefix (LLaVA paradigm) prepends visual tokens to text input
       → BioGPT's existing self-attention handles everything
       → no missing weights, no architectural hacks
       → proven approach used by LLaVA, MiniGPT-4, InstructBLIP
"""

import torch
import torch.nn as nn
import clip
from transformers import BioGptForCausalLM, BioGptTokenizer

from config import (
    CLIP_MODEL_NAME, CLIP_EMBED_DIM, CLIP_HIDDEN_DIM, CLIP_NUM_PATCHES,
    DECODER_MODEL_NAME, DECODER_HIDDEN_DIM,
    NUM_CLASSES, PATHOLOGY_CLASSES,
    CLASSIFIER_CFG, DECODER_CFG,
)


class MedicalVLM(nn.Module):
    """
    Full Vision-Language Model for medical chest X-ray analysis.

    Components:
        1. CLIP ViT-B/32 visual encoder (pre-trained on 400M image-text pairs)
        2. 2-layer MLP classification head (512 → 13 classes)
        3. Bridge MLP: Linear(768→1024) + LayerNorm + GELU + Dropout
        4. BioGPT decoder (347M params, pre-trained on 15M PubMed abstracts)

    Visual Prefix Injection (LLaVA paradigm):
        Instead of cross-attention (which was broken in the old pipeline),
        we prepend 49 bridged visual tokens to the text token sequence.
        BioGPT's self-attention naturally attends to both visual and text tokens.

        Input to BioGPT: [visual_0, visual_1, ..., visual_48, text_0, text_1, ..., text_T]
        Loss computed only on text tokens (visual prefix is -100 in labels).
    """

    def __init__(self, device: torch.device, freeze_encoder: bool = True):
        super().__init__()
        self.device = device
        self._patch_features = None

        # ── 1. CLIP Visual Encoder ────────────────────────────────
        clip_model, _ = clip.load(CLIP_MODEL_NAME, device='cpu')
        self.visual = clip_model.visual

        # Freeze all, then selectively unfreeze
        for param in self.visual.parameters():
            param.requires_grad_(False)

        if not freeze_encoder:
            for block in self.visual.transformer.resblocks[-2:]:
                for param in block.parameters():
                    param.requires_grad_(True)

        # Hook to capture 49 patch tokens from last ViT block
        self._hook = self.visual.transformer.resblocks[-1].register_forward_hook(
            self._patch_hook
        )

        # ── 2. Classification Head ────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(CLIP_EMBED_DIM),
            nn.Linear(CLIP_EMBED_DIM, CLIP_EMBED_DIM),
            nn.GELU(),
            nn.Dropout(CLASSIFIER_CFG['dropout']),
            nn.Linear(CLIP_EMBED_DIM, NUM_CLASSES),
        )

        # ── 3. Bridge MLP ─────────────────────────────────────────
        # Projects ViT patch tokens (768-dim) to BioGPT embedding space (1024-dim)
        self.bridge = nn.Sequential(
            nn.Linear(CLIP_HIDDEN_DIM, DECODER_HIDDEN_DIM),
            nn.LayerNorm(DECODER_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # ── 4. BioGPT Decoder ─────────────────────────────────────
        self.decoder = BioGptForCausalLM.from_pretrained(DECODER_MODEL_NAME)
        self.decoder.config.tie_word_embeddings = False

        # Print architecture summary
        self._print_summary()

    def _print_summary(self):
        enc_total = sum(p.numel() for p in self.visual.parameters())
        enc_train = sum(p.numel() for p in self.visual.parameters() if p.requires_grad)
        cls_params = sum(p.numel() for p in self.classifier.parameters())
        bridge_params = sum(p.numel() for p in self.bridge.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total = enc_total + cls_params + bridge_params + dec_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n[VLM] Architecture Summary:")
        print(f"  CLIP ViT-B/32:     {enc_total:>12,} params ({enc_train:,} trainable)")
        print(f"  Classifier head:   {cls_params:>12,} params")
        print(f"  Bridge MLP:        {bridge_params:>12,} params")
        print(f"  BioGPT decoder:    {dec_params:>12,} params (pre-trained on PubMed)")
        print(f"  Total:             {total:>12,} params ({trainable:,} trainable)")

    # ── Patch Token Hook ──────────────────────────────────────────

    def _patch_hook(self, module, input, output):
        """
        Captures 49 spatial patch tokens from the last ViT transformer block.
        ViT output: (seq_len=50, batch, dim=768) — position 0 is CLS, 1-49 are patches.
        """
        self._patch_features = output[1:, :, :].permute(1, 0, 2).contiguous()  # (B, 49, 768)
        if self._patch_features.requires_grad:
            self._patch_features.retain_grad()

    # ── Encoding Methods ──────────────────────────────────────────

    def encode_image_cls(self, images: torch.Tensor) -> torch.Tensor:
        """CLS embedding for classification. Shape: (B, 512)"""
        return self.visual(images.float())

    def encode_image_patches(self, images: torch.Tensor, enable_grads: bool = False) -> torch.Tensor:
        """
        Patch tokens through bridge for decoder input. Shape: (B, 49, 1024)
        These become the visual prefix in the decoder input sequence.
        """
        if enable_grads:
            _ = self.visual(images.float())
        else:
            with torch.no_grad():
                _ = self.visual(images.float())
        return self.bridge(self._patch_features)

    # ── Classification ────────────────────────────────────────────

    def classify(self, images: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (B, NUM_CLASSES). Apply sigmoid for probabilities."""
        cls_features = self.encode_image_cls(images)
        return self.classifier(cls_features.float())

    # ── Report Generation (Training) ──────────────────────────────

    def forward_decoder(self, images, input_ids, attention_mask, labels=None):
        """
        Forward pass for report generation training using visual prefix injection.

        1. Encode image → 49 visual tokens → bridge → (B, 49, 1024)
        2. Embed text tokens → (B, T, 1024)
        3. Concatenate: [visual_prefix | text_tokens] → (B, 49+T, 1024)
        4. Create combined attention mask
        5. Labels: -100 for visual positions (no loss on visual tokens)
        6. Forward through BioGPT decoder
        """
        B = images.shape[0]

        # Visual prefix
        visual_embeds = self.encode_image_patches(images)         # (B, 49, 1024)

        # Text embeddings (with BioGPT's learned scaling)
        text_embeds = self.decoder.biogpt.embed_tokens(input_ids) # (B, T, 1024)

        # Concatenate: [visual | text]
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)  # (B, 49+T, 1024)

        # Combined attention mask
        visual_mask = torch.ones(B, CLIP_NUM_PATCHES,
                                 dtype=attention_mask.dtype,
                                 device=self.device)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)  # (B, 49+T)

        # Labels: -100 for visual prefix positions (ignored by loss)
        if labels is not None:
            visual_labels = torch.full((B, CLIP_NUM_PATCHES), -100,
                                        dtype=labels.dtype,
                                        device=self.device)
            combined_labels = torch.cat([visual_labels, labels], dim=1)  # (B, 49+T)
        else:
            combined_labels = None

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

    # ── Report Generation (Inference) ─────────────────────────────

    @torch.no_grad()
    def generate_report(
        self,
        image: torch.Tensor,
        tokenizer: BioGptTokenizer,
        max_length: int = None,
        prefix: str = "Findings:",
    ) -> str:
        """
        Generates a radiology report for a single image.

        Uses autoregressive decoding with the visual prefix prepended.
        At each step, BioGPT's self-attention attends to both the 49
        visual tokens and all previously generated text tokens.
        """
        self.eval()
        if max_length is None:
            max_length = DECODER_CFG['gen_max_length']

        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Visual prefix
        visual_embeds = self.encode_image_patches(image)  # (1, 49, 1024)

        # Encode prefix text
        prefix_ids = tokenizer.encode(prefix, return_tensors='pt').to(self.device)

        # Autoregressive generation with visual prefix
        generated_ids = prefix_ids.clone()

        for _ in range(max_length):
            # Get text embeddings for current sequence
            text_embeds = self.decoder.biogpt.embed_tokens(generated_ids)

            # Prepend visual prefix
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)

            # Forward
            outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            # Get next token logits (last position)
            next_logits = outputs.logits[0, -1, :]

            # Apply repetition penalty
            for token_id in generated_ids[0]:
                next_logits[token_id] /= 1.2

            # Greedy decode (or top-k sampling)
            next_token = torch.argmax(next_logits, dim=-1)

            # Stop on EOS
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
        Falls back to greedy decoding if beam search fails.
        """
        self.eval()
        if max_new_tokens is None:
            max_new_tokens = DECODER_CFG['gen_max_length']

        if image.dim() == 3:
            image = image.unsqueeze(0)

        visual_embeds = self.encode_image_patches(image)
        prefix_ids = tokenizer.encode(prefix, return_tensors='pt').to(self.device)
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

    # ── Phase Control Methods ─────────────────────────────────────

    def freeze_for_classification(self):
        """Phase 1: Classifier head + top N encoder blocks trainable."""
        for p in self.parameters():
            p.requires_grad_(False)

        for p in self.classifier.parameters():
            p.requires_grad_(True)

        n_blocks = CLASSIFIER_CFG.get('unfreeze_blocks', 2)
        for block in self.visual.transformer.resblocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VLM] Classification mode: {trainable:,} trainable params")

    def freeze_for_decoder_prewarm(self):
        """
        Phase 2a: Only bridge MLP trainable.
        This forces the bridge to learn the mapping from ViT visual space
        to BioGPT text space before the decoder starts adjusting.
        Much simpler than the old approach of trying to find cross-attention
        layers — here we just train the bridge.
        """
        for p in self.parameters():
            p.requires_grad_(False)

        for p in self.bridge.parameters():
            p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VLM] Prewarm mode: {trainable:,} trainable params (bridge only)")

    def freeze_for_decoder_full(self):
        """Phase 2b: Bridge + full BioGPT decoder trainable. Encoder frozen."""
        for p in self.parameters():
            p.requires_grad_(False)

        for p in self.bridge.parameters():
            p.requires_grad_(True)

        for p in self.decoder.parameters():
            p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[VLM] Full decoder mode: {trainable:,} trainable params")

    def get_classifier_param_groups(self):
        """Differential learning rates: slow backbone, faster head."""
        n_blocks = CLASSIFIER_CFG.get('unfreeze_blocks', 2)
        backbone_params = []
        for block in self.visual.transformer.resblocks[-n_blocks:]:
            backbone_params.extend(list(block.parameters()))

        return [
            {'params': backbone_params, 'lr': CLASSIFIER_CFG['lr_backbone']},
            {'params': list(self.classifier.parameters()), 'lr': CLASSIFIER_CFG['lr_head']},
        ]

