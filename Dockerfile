# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip \
    ffmpeg libsm6 libxext6 \
    git wget

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone --branch v0.2.3 https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade -r requirements.txt

# Install ComfyUI-Manager - handy for debugging. Should probably remove for production
# RUN git clone https://github.com/ltdrdata/ComfyUI-Manager.git /comfyui/custom_nodes/ComfyUI-Manager
# RUN pip3 install -r /comfyui/custom_nodes/ComfyUI-Manager/requirements.txt

# Install ComfyUI-Impact-Pack
RUN git clone --branch 7.10.6 https://github.com/ltdrdata/ComfyUI-Impact-Pack.git /comfyui/custom_nodes/ComfyUI-Impact-Pack
RUN pip3 install -r /comfyui/custom_nodes/ComfyUI-Impact-Pack/requirements.txt
RUN python3 /comfyui/custom_nodes/ComfyUI-Impact-Pack/install.py

# Install comfyui_controlnet_aux
RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git /comfyui/custom_nodes/comfyui_controlnet_aux
RUN pip3 install --upgrade -r /comfyui/custom_nodes/comfyui_controlnet_aux/requirements.txt  
RUN mkdir -p /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/depth-anything/Depth-Anything-V2-Large
ADD data/depth_anything_v2_vitl.pth /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/depth-anything/Depth-Anything-V2-Large/
#RUN wget -O /comfyui/custom_nodes/comfyui_controlnet_aux/ckpts/depth-anything/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth \
#    https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

# install was-node-suite-comfyui
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git /comfyui/custom_nodes/was-node-suite-comfyui
RUN git -C /comfyui/custom_nodes/was-node-suite-comfyui checkout 9cb63d3f0f576d9023e9295b84ad9cd115ce69dc
RUN pip3 install -r /comfyui/custom_nodes/was-node-suite-comfyui/requirements.txt

# Install ComfyUI-Inspyrenet-Rembg
RUN git clone https://github.com/john-mnz/ComfyUI-Inspyrenet-Rembg.git /comfyui/custom_nodes/ComfyUI-Inspyrenet-Rembg
RUN git -C /comfyui/custom_nodes/ComfyUI-Inspyrenet-Rembg checkout 87ac452ef1182e8f35f59b04010158d74dcefd06
RUN pip3 install -r /comfyui/custom_nodes/ComfyUI-Inspyrenet-Rembg/requirements.txt

# install ComfyUI_IPAdapter_plus
RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git /comfyui/custom_nodes/ComfyUI_IPAdapter_plus
RUN git -C /comfyui/custom_nodes/ComfyUI_IPAdapter_plus checkout b188a6cb39b512a9c6da7235b880af42c78ccd0d

# Install comfyui_various
RUN git clone https://github.com/jamesWalker55/comfyui-various.git /comfyui/custom_nodes/comfyui-various
RUN git -C /comfyui/custom_nodes/comfyui-various checkout 36454f91606bbff4fc36d90234981ca4a47e2695

# Install ComfyLiterals
RUN git clone https://github.com/M1kep/ComfyLiterals.git /comfyui/custom_nodes/ComfyLiterals
RUN git -C /comfyui/custom_nodes/ComfyLiterals checkout bdddb08ca82d90d75d97b1d437a652e0284a32ac

# Install ComfyUI_essentials
RUN git clone https://github.com/cubiq/ComfyUI_essentials.git /comfyui/custom_nodes/ComfyUI_essentials
RUN git -C /comfyui/custom_nodes/ComfyUI_essentials checkout 64e38fd0f3b2e925573684f4a43727be80dc7d5b
RUN pip3 install -r /comfyui/custom_nodes/ComfyUI_essentials/requirements.txt

# Install ComfyUI-KJNodes
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git /comfyui/custom_nodes/ComfyUI-KJNodes
RUN git -C /comfyui/custom_nodes/ComfyUI-KJNodes checkout d9191b4c1d6b738bd35c62cc6de51c32f0a5b158
RUN pip3 install -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt

# Install ComfyUI-IC-Light
RUN git clone https://github.com/kijai/ComfyUI-IC-Light.git /comfyui/custom_nodes/ComfyUI-IC-Light
RUN git -C /comfyui/custom_nodes/ComfyUI-IC-Light checkout 8a9f9c92c155e754a05840c7621443c5919a9b25

# Install https://github.com/storyicon/comfyui_segment_anything.git
RUN git clone https://github.com/storyicon/comfyui_segment_anything.git /comfyui/custom_nodes/comfyui_segment_anything
RUN git -C /comfyui/custom_nodes/comfyui_segment_anything checkout ab6395596399d5048639cdab7e44ec9fae857a93
RUN pip3 install --upgrade -r /comfyui/custom_nodes/comfyui_segment_anything/requirements.txt

# Install comfyui_face_parsing
RUN git clone https://github.com/Ryuukeisyou/comfyui_face_parsing.git /comfyui/custom_nodes/comfyui_face_parsing
RUN pip3 install -r /comfyui/custom_nodes/comfyui_face_parsing/requirements.txt
RUN wget -P /comfyui/models/face_parsing/ \
    https://huggingface.co/jonathandinu/face-parsing/resolve/main/model.safetensors \
    https://huggingface.co/jonathandinu/face-parsing/resolve/main/quantize_config.json \
    https://huggingface.co/jonathandinu/face-parsing/resolve/main/preprocessor_config.json \
    https://huggingface.co/jonathandinu/face-parsing/resolve/main/config.json

# Install comfyui_face_detection
RUN git clone https://github.com/risunobushi/comfyUI_FrequencySeparation_RGB-HSV.git /comfyui/custom_nodes/comfyUI_FrequencySeparation_RGB-HSV
RUN git -C /comfyui/custom_nodes/comfyUI_FrequencySeparation_RGB-HSV checkout 67a08c55ee6aa8e9140616f01497bd54d3533fa6

# https://github.com/storyicon/comfyui_segment_anything/issues/88
RUN pip3 install 'timm==1.0.9' 


# Copy some images for testing
ADD test_resources/images/fockers.jpg test_resources/images/brad.jpg test_resources/images/ws.jpg /comfyui/input/


# Install runpod
RUN pip3 install runpod requests boto3 uuid

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh


# Stage 2: Download models
#FROM base AS downloader

# Copy the local comfyui folder to a temporary location on the image
#COPY ./comfyui /tmp/comfyui

# Merge the folders using rsync
#RUN rsync -a /tmp/comfyui/ /comfyui 

# Remove the tmp copy
#RUN rm -rf /tmp/comfyui

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

# Stage 3: Final image
#FROM base AS final

# Copy models from stage 2 to the final image
#COPY --from=downloader /comfyui/models /comfyui/models

# Start the container
CMD ["/start.sh"]