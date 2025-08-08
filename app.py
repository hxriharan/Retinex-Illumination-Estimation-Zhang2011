import gradio as gr
import cv2
import numpy as np
from PIL import Image
from improved_retinex import run_irr_pipeline
from ssr_msr import single_scale_retinex, multi_scale_retinex

# -------- utils (inline to avoid extra imports) --------
def to_rgb_uint8(img_pil: Image.Image) -> np.ndarray:
    """Ensure 3-channel RGB uint8 numpy array."""
    if img_pil.mode == "RGBA":
        img_pil = img_pil.convert("RGB")
    elif img_pil.mode == "L":
        img_pil = img_pil.convert("RGB")
    return np.array(img_pil).astype(np.uint8)

def to_pil(img_np: np.ndarray) -> Image.Image:
    """Clamp/convert numpy array to PIL (RGB uint8)."""
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    return Image.fromarray(img_np, mode="RGB")

# -------- core processing --------
def process_image_all_algorithms(image: Image.Image):
    if image is None:
        return None, None, None, None

    img_rgb = to_rgb_uint8(image)

    try:
        # SSR (expects uint8 RGB, returns float [0,1] in your impl)
        ssr = single_scale_retinex(img_rgb, sigma=80)
        ssr_u8 = (np.clip(ssr, 0, 1) * 255).astype(np.uint8)

        # MSR
        msr = multi_scale_retinex(img_rgb, sigmas=[15, 80, 250])
        msr_u8 = (np.clip(msr, 0, 1) * 255).astype(np.uint8)

        # IRR — CHANGED: keep params consistent with what you wanted
        irr = run_irr_pipeline(
            img_rgb,
            alpha=0.05,            # CHANGED
            threshold=0.4,         # CHANGED
            enhance_method='normalize'
        )["enhanced"]

        return (
            to_pil(img_rgb),
            to_pil(ssr_u8),
            to_pil(msr_u8),
            to_pil(irr),
        )
    except Exception as e:
        # Return originals so UI doesn't blank out; print for logs
        print(f"Error processing image: {e}")
        img_pil = to_pil(img_rgb)
        return img_pil, img_pil, img_pil, img_pil

# -------- UI --------
def create_ui():
    with gr.Blocks(title="Recursive Retinex Implementation (Zhang 2011)", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Recursive Retinex Implementation (Zhang 2011)")
        gr.Markdown(
            "Upload an image to see enhanced results using Single Scale Retinex (SSR), "
            "Multi Scale Retinex (MSR), and Improved Recursive Retinex (IRR)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Image")
                input_image = gr.Image(label="Input Image", type="pil")

                # Optional: live-tunable IRR params
                with gr.Accordion("IRR Parameters", open=False):
                    alpha = gr.Slider(0.01, 0.2, value=0.05, step=0.005, label="alpha (smoothing)")
                    thr   = gr.Slider(0.05, 3.0, value=0.4, step=0.05, label="threshold (log units)")

                process_btn = gr.Button("Process Image", variant="primary")

                gr.Markdown("### Algorithm Information")
                gr.Markdown(
                    "**SSR (Single Scale Retinex)**: Gaussian blur, sigma=80  \n"
                    "**MSR (Multi Scale Retinex)**: scales [15, 80, 250]  \n"
                    "**IRR (Improved Recursive Retinex)**: 8-direction recursion with "
                    "**α=0.05**, **threshold=0.4** (log units)"  # CHANGED
                )

            with gr.Column(scale=4):
                gr.Markdown("### Results")
                with gr.Row():
                    output_original = gr.Image(label="Original Image", type="pil")
                    output_ssr      = gr.Image(label="SSR Enhanced", type="pil")
                    output_msr      = gr.Image(label="MSR Enhanced", type="pil")
                    output_irr      = gr.Image(label="IRR Enhanced", type="pil")

        # --- handlers ---
        def process_image(image, a, t):
            # Pipe sliders into IRR while keeping SSR/MSR constant
            if image is None:
                return None, None, None, None

            img_rgb = to_rgb_uint8(image)

            ssr = single_scale_retinex(img_rgb, sigma=80)
            msr = multi_scale_retinex(img_rgb, sigmas=[15, 80, 250])
            irr = run_irr_pipeline(img_rgb, alpha=a, threshold=t, enhance_method='normalize')["enhanced"]

            return (
                to_pil(img_rgb),
                to_pil((np.clip(ssr, 0, 1) * 255).astype(np.uint8)),
                to_pil((np.clip(msr, 0, 1) * 255).astype(np.uint8)),
                to_pil(irr),
            )

        inputs = [input_image, alpha, thr]
        outputs = [output_original, output_ssr, output_msr, output_irr]

        process_btn.click(fn=process_image, inputs=inputs, outputs=outputs)
        input_image.change(fn=process_image, inputs=inputs, outputs=outputs)

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True, debug=True)
