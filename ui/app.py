import gradio as gr
import numpy as np
import cv2
from retinex.gradients import compute_gradients, visualize_gradients
from retinex.illumination import estimate_illumination
import matplotlib.pyplot as plt
from PIL import Image
import io

# Utility: convert matplotlib fig to image for display
def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

# Option 1: Log RGB Gradients
def show_gradients(img):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    log_img = np.log1p(rgb.astype(np.float32) / 255.0)
    grad_x, grad_y = compute_gradients(log_img)
    grad_mag = visualize_gradients(grad_x, grad_y)

    fig, ax = plt.subplots()
    ax.imshow(grad_mag, cmap='inferno')
    ax.set_title("Log RGB Gradient Magnitude")
    ax.axis('off')
    return fig_to_image(fig)

# Option 2: Illumination Estimation
def show_illumination(img, method="gaussian", kernel_size=15, sigma=5, d=31, sigmaColor=100, sigmaSpace=20):
    image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    illum = estimate_illumination(image, method, kernel_size, sigma, d, sigmaColor, sigmaSpace)

    fig, ax = plt.subplots()
    ax.imshow(illum)
    ax.set_title(f"Illumination Map ({method})")
    ax.axis('off')
    return fig_to_image(fig)

# Combined function for Gradio
def process_image(image, option, method, ksize, sigma, d, sigColor, sigSpace):
    if option == "Log Gradient":
        return show_gradients(image)
    elif option == "Illumination":
        return show_illumination(image, method, ksize, sigma, d, sigColor, sigSpace)
    else:
        return image  # fallback

# Gradio UI
def main_ui():
    with gr.Blocks() as demo:
        gr.Markdown("## Retinex Demo: Gradient and Illumination Estimation")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
            image_output = gr.Image(type="pil", label="Processed Output")

        option = gr.Radio(["Log Gradient", "Illumination"], label="Select Operation", value="Log Gradient")
        method = gr.Dropdown(["gaussian", "bilateral"], label="Smoothing Method", value="gaussian")
        ksize = gr.Slider(3, 51, step=2, value=15, label="Kernel Size (Gaussian Only)")
        sigma = gr.Slider(1, 30, step=1, value=5, label="Sigma (Gaussian Only)")

        with gr.Group(visible=False) as bilateral_controls:
            d = gr.Slider(3, 51, step=2, value=31, label="d (Neighborhood Size)")
            sigColor = gr.Slider(1, 200, step=1, value=100, label="Sigma Color")
            sigSpace = gr.Slider(1, 200, step=1, value=20, label="Sigma Space")

        # Toggle bilateral parameter visibility based on method
        def toggle_bilateral(m):
            return gr.update(visible=(m == "bilateral"))

        method.change(fn=toggle_bilateral, inputs=method, outputs=bilateral_controls)

        run_btn = gr.Button("Run")

        run_btn.click(
            fn=process_image,
            inputs=[image_input, option, method, ksize, sigma, d, sigColor, sigSpace],
            outputs=image_output
        )

    return demo

if __name__ == "__main__":
    ui = main_ui()
    ui.launch()
