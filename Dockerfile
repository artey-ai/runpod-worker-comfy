# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    rsync

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade -r requirements.txt

# Install ComfyUI-Manager
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /comfyui/custom_nodes/ComfyUI-Manager
RUN pip3 install -r /comfyui/custom_nodes/ComfyUI-Manager/requirements.txt

# Install ComfyUI-Impact-Pack
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /comfyui/custom_nodes/ComfyUI-Impact-Pack
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack /comfyui/custom_nodes/ComfyUI-Impact-Pack/impact_subpack
RUN pip3 install -r /comfyui/custom_nodes/ComfyUI-Impact-Pack/requirements.txt
RUN python3 /comfyui/custom_nodes/ComfyUI-Impact-Pack/install.py

# Install comfyui_controlnet_aux
RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git /comfyui/custom_nodes/comfyui_controlnet_aux
RUN pip3 install -r /comfyui/custom_nodes/comfyui_controlnet_aux/requirements.txt

# Install comfyui_face_parsing
RUN git clone https://github.com/Ryuukeisyou/comfyui_face_parsing.git /comfyui/custom_nodes/comfyui_face_parsing
RUN pip3 install -r /comfyui/custom_nodes/comfyui_face_parsing/requirements.txt

# install ComfyUI-SAM2
RUN git clone https://github.com/neverbiasu/ComfyUI-SAM2.git /comfyui/custom_nodes/ComfyUI-SAM2
RUN pip3 install -r /comfyui/custom_nodes/ComfyUI-SAM2/requirements.txt

# install was-node-suite-comfyui
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git /comfyui/custom_nodes/was-node-suite-comfyui
RUN pip3 install -r /comfyui/custom_nodes/was-node-suite-comfyui/requirements.txt

# Install ComfyUI_essentials
RUN git clone https://github.com/cubiq/ComfyUI_essentials.git /comfyui/custom_nodes/ComfyUI_essentials
RUN pip3 install -r /comfyui/custom_nodes/ComfyUI_essentials/requirements.txt

# Install ComfyUI-KJNodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes
RUN pip3 install -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt

# Install ComfyUI-Crystools
RUN git clone https://github.com/crystian/ComfyUI-Crystools.git /comfyui/custom_nodes/ComfyUI-Crystools
RUN pip3 install -r /comfyui/custom_nodes/ComfyUI-Crystools/requirements.txt

# Clone other custom nodes (with no requirements)
RUN git clone https://github.com/kijai/ComfyUI-IC-Light.git /comfyui/custom_nodes/ComfyUI-IC-Light
RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git /comfyui/custom_nodes/ComfyUI_IPAdapter_plus
RUN git clone https://github.com/sipherxyz/comfyui-art-venture.git /comfyui/custom_nodes/comfyui-art-venture
RUN git clone https://github.com/jamesWalker55/comfyui-various.git /comfyui/custom_nodes/comfyui-various
RUN git clone https://github.com/rgthree/rgthree-comfy.git /comfyui/custom_nodes/rgthree-comfy
RUN git clone https://github.com/risunobushi/comfyUI_FrequencySeparation_RGB-HSV.git /comfyui/custom_nodes/comfyUI_FrequencySeparation_RGB-HSV

# Copy some images for testing
ADD test_resources/images/bear.jpg test_resources/images/brad.jpg test_resources/images/ws.jpg /comfyui/input/



# Install runpod
RUN pip3 install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh


# Copy the local folder to a temporary location
COPY ./comfyui /tmp/comfyui
# Merge the folders using rsync
RUN rsync -a /tmp/comfyui/ /comfyui && rm -rf /tmp/comfyui


# # Stage 2: Download models
# FROM base as downloader

# # Change working directory to ComfyUI
# WORKDIR /comfyui

# # Download checkpoints/vae/LoRA to include in image based on model type
# RUN if [ "$MODEL_TYPE" = "sdxl" ]; then \
#       wget -O models/checkpoints/sd_xl_base_1.0.safetensors https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors && \
#       wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors && \
#       wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors; \
#     elif [ "$MODEL_TYPE" = "sd3" ]; then \
#       wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/checkpoints/sd3_medium_incl_clips_t5xxlfp8.safetensors https://huggingface.co/stabilityai/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors; \
#     elif [ "$MODEL_TYPE" = "flux1-schnell" ]; then \
#       wget -O models/unet/flux1-schnell.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors && \
#       wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
#       wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
#       wget -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors; \
#     elif [ "$MODEL_TYPE" = "flux1-dev" ]; then \
#       wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors && \
#       wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors && \
#       wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors && \
#       wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors; \
#     fi

# # Stage 3: Final image
# FROM base as final

# # Copy models from stage 2 to the final image
# COPY --from=downloader /comfyui/models /comfyui/models

# Start the container
CMD /start.sh