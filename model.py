"""
model.py — Vision-language model with CLIP visual encoder + BioGPT decoder
"""

import torch
import torch.nn as nn
import clip
from transformers import BioGptForCausalLM, BioGptTokenizer

from config import (
    CLASSIFIER_CFG,
    CLIP_EMBED_DIM,
    CLIP_HIDDEN_DIM,
    CLIP_MODEL_NAME,
    CLIP_NUM_PATCHES,
    DECODER_CFG,
    DECODER_HIDDEN_DIM,
    DECODER_MODEL_NAME,
    NUM_CLASSES,
)


class MedicalVLM(nn.Module):
    """Student-scale medical VLM for chest X-ray classification and report generation."""

    def __init__(self, device: torch.device, freeze_encoder: bool = True):
        super().__init__()
        self.device = device
        self._patch_features = None

        clip_model, _ = clip.load(CLIP_MODEL_NAME, device='cpu')
        self.visual = clip_model.visual

        for param in self.visual.parameters():
            param.requires_grad_(False)

        if not freeze_encoder:
            for block in self.visual.transformer.resblocks[-2:]:
                for param in block.parameters():
                    param.requires_grad_(True)

        self._hook = self.visual.transformer.resblocks[-1].register_forward_hook(self._patch_hook)

        self.classifier = nn.Sequential(
            nn.LayerNorm(CLIP_EMBED_DIM),
            nn.Linear(CLIP_EMBED_DIM, CLIP_EMBED_DIM),
            nn.GELU(),
            nn.Dropout(CLASSIFIER_CFG['dropout']),
            nn.Linear(CLIP_EMBED_DIM, NUM_CLASSES),
        )

        self.bridge = nn.Sequential(
            nn.Linear(CLIP_HIDDEN_DIM, DECODER_HIDDEN_DIM),
            nn.LayerNorm(DECODER_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.decoder = BioGptForCausalLM.from_pretrained(DECODER_MODEL_NAME)
        self._print_summary()

    def _print_summary(self):
        enc_total = sum(p.numel() for p in self.visual.parameters())
        enc_train = sum(p.numel() for p in self.visual.parameters() if p.requires_grad)
        cls_params = sum(p.numel() for p in self.classifier.parameters())
        bridge_params = sum(p.numel() for p in self.bridge.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total = enc_total + cls_params + bridge_params + dec_params
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('\n[VLM] Architecture Summary:')
        print(f'  CLIP ViT-B/32:   {enc_total:>12,} params ({enc_train:,} trainable)')
        print(f'  Classifier head: {cls_params:>12,} params')
        print(f'  Bridge MLP:      {bridge_params:>12,} params')
        print(f'  BioGPT decoder:  {dec_params:>12,} params')
        print(f'  Total:           {total:>12,} params ({trainable:,} trainable)')

    def _patch_hook(self, module, inputs, output):
        patch_tokens = output[1:, :, :].permute(1, 0, 2)  # (B, 49, 768)
        if patch_tokens.requires_grad:
            patch_tokens.retain_grad()
        self._patch_features = patch_tokens

    def encode_image_cls(self, images: torch.Tensor) -> torch.Tensor:
        return self.visual(images.float())

    def encode_image_patches(self, images: torch.Tensor) -> torch.Tensor:
        _ = self.visual(images.float())
        if self._patch_features is None:
            raise RuntimeError('Patch features were not captured by the forward hook.')
        return self.bridge(self._patch_features)

    def classify(self, images: torch.Tensor) -> torch.Tensor:
        cls_features = self.encode_image_cls(images)
        return self.classifier(cls_features.float())

    def forward_decoder(self, images, input_ids, attention_mask, labels=None):
        batch_size = images.shape[0]
        visual_embeds = self.encode_image_patches(images)
        text_embeds = self.decoder.biogpt.embed_tokens(input_ids)
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        visual_mask = torch.ones(batch_size, CLIP_NUM_PATCHES,
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
        combined_mask = torch.cat([visual_mask, attention_mask], dim=1)

        if labels is not None:
            visual_labels = torch.full((batch_size, CLIP_NUM_PATCHES), -100,
                                       dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([visual_labels, labels], dim=1)
        else:
            combined_labels = None

        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

    @torch.no_grad()
    def generate_report(self, image: torch.Tensor, tokenizer: BioGptTokenizer,
                        max_length: int | None = None) -> str:
        self.eval()
        if max_length is None:
            max_length = DECODER_CFG['gen_max_length']

        if image.dim() == 3:
            image = image.unsqueeze(0)

        visual_embeds = self.encode_image_patches(image)
        prefix_ids = tokenizer.encode('Findings:', return_tensors='pt').to(self.device)
        generated_ids = prefix_ids.clone()

        for _ in range(max_length):
            text_embeds = self.decoder.biogpt.embed_tokens(generated_ids)
            inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=self.device)
            outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            next_logits = outputs.logits[0, -1, :]
            for token_id in generated_ids[0]:
                next_logits[token_id] /= 1.2
            next_token = torch.argmax(next_logits, dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_ids = torch.cat([generated_ids, next_token.view(1, 1)], dim=1)

        return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def freeze_for_classification(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.classifier.parameters():
            p.requires_grad_(True)
        for block in self.visual.transformer.resblocks[-2:]:
            for p in block.parameters():
                p.requires_grad_(True)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[VLM] Classification mode: {trainable:,} trainable params')

    def freeze_for_decoder_prewarm(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.bridge.parameters():
            p.requires_grad_(True)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[VLM] Decoder prewarm mode: {trainable:,} trainable params')

    def freeze_for_decoder_full(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.bridge.parameters():
            p.requires_grad_(True)
        for p in self.decoder.parameters():
            p.requires_grad_(True)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[VLM] Full decoder mode: {trainable:,} trainable params')

    def get_classifier_param_groups(self):
        backbone_params = []
        for block in self.visual.transformer.resblocks[-2:]:
            backbone_params.extend(list(block.parameters()))
        return [
            {'params': backbone_params, 'lr': CLASSIFIER_CFG['lr_backbone']},
            {'params': list(self.classifier.parameters()), 'lr': CLASSIFIER_CFG['lr_head']},
        ]
