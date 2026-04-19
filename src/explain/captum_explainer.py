"""Captum model explainer for FraudLens."""

import base64
import io
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerIntegratedGradients

matplotlib.use("Agg")
logger = logging.getLogger(__name__)


class CaptumExplainer:
    """Generates deep multimodal explanations using Captum."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def explain(self, tab, img, txt_ids, txt_mask, pil_image=None, tokens=None, img_score=100.0):
        """Run attributions for text tokens and image pixels."""
        out = {"text_attributions": [], "image_explanation_base64": None}
        self.model.zero_grad()

        # ---------------------------------------------------------------------
        # 1. Text Attributions (Layer Integrated Gradients on word_embeddings)
        # ---------------------------------------------------------------------
        if txt_ids is not None and tokens is not None:
            def custom_txt_fwd(input_ids_, tab_, img_, mask_):
                return self.model(tabular=tab_, image=img_, input_ids=input_ids_, attention_mask=mask_)["probability"]

            try:
                # Target the distilbert word_embeddings layer
                target_layer = self.model.text_branch.bert.embeddings.word_embeddings
                lig_text = LayerIntegratedGradients(custom_txt_fwd, target_layer)

                attrs_txt, _ = lig_text.attribute(
                    inputs=txt_ids,
                    additional_forward_args=(tab, img, txt_mask),
                    n_steps=15,
                    internal_batch_size=1,
                    return_convergence_delta=True
                )
                
                # Sum across embedding dimensions
                attrs_txt = attrs_txt.sum(dim=-1).squeeze(0).cpu().detach().numpy()  # (seq_len,)
                
                text_result = []
                for i, token in enumerate(tokens):
                    if token not in ["[CLS]", "[SEP]", "[PAD]", ""]:
                        # Exclude ## subwords by stripping later or just keeping them
                        wt = float(attrs_txt[i])
                        # Keep ## prefix to merge or highlight gracefully
                        text_result.append({"word": token, "weight": wt})
                
                # Normalize weights between -1 and 1
                if text_result:
                    max_abs = max(abs(x["weight"]) for x in text_result)
                    if max_abs > 0:
                        for x in text_result:
                            x["weight"] = x["weight"] / max_abs

                out["text_attributions"] = text_result
            except Exception as e:
                logger.error("Text explainer failed: %s", e)

        # ---------------------------------------------------------------------
        # 2. Image Attributions (Integrated Gradients on Pixels)
        # ---------------------------------------------------------------------
        if pil_image is not None and img is not None and img.sum().item() != 0:
            def custom_img_fwd(img_, tab_, txt_ids_, mask_):
                return self.model(tabular=tab_, image=img_, input_ids=txt_ids_, attention_mask=mask_)["probability"]

            try:
                ig_image = IntegratedGradients(custom_img_fwd)
                
                attrs_img, _ = ig_image.attribute(
                    inputs=img,
                    additional_forward_args=(tab, txt_ids, txt_mask),
                    n_steps=15,
                    internal_batch_size=1,
                    return_convergence_delta=True
                )
                
                attrs_img = attrs_img.squeeze(0).cpu().detach().numpy() # (3, 224, 224)
                
                # Average across channels and take absolute value
                heatmap = np.mean(attrs_img, axis=0)
                heatmap = np.maximum(heatmap, 0) # Only positive contributions
                
                # Apply Gaussian filter to smooth the noisy Integrated Gradients
                from scipy.ndimage import gaussian_filter
                heatmap = gaussian_filter(heatmap, sigma=4)
                
                if heatmap.max() > 0:
                    heatmap = heatmap / heatmap.max()
                
                # Generate overlay image
                import PIL.Image
                orig = pil_image.resize((224, 224), PIL.Image.Resampling.BILINEAR)
                orig_np = np.array(orig)
                
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(orig_np)
                
                # Mask out low values so they are completely transparent
                masked_heatmap = np.ma.masked_where(heatmap < 0.15, heatmap)
                
                # Adjust opacity based on severity
                alpha_val = 0.6 if img_score > 70 else 0.4
                ax.imshow(masked_heatmap, cmap='jet', alpha=alpha_val)
                
                ax.axis('off')
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
                plt.close(fig)
                buf.seek(0)
                encoded = base64.b64encode(buf.read()).decode('utf-8')
                out["image_explanation_base64"] = "data:image/png;base64," + encoded
                
            except Exception as e:
                import traceback
                logger.error("Image explainer failed: %s\n%s", e, traceback.format_exc())

        return out
