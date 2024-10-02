
1. Install Ubuntu from the Windows store
2. Launch Ubuntu
3. sudo apt-get update && sudo apt-get -y upgrade
4. Install Docker: https://docs.docker.com/engine/install/ubuntu/
5. Install NVIDIA Toolkit for Docker: 
    - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation
    - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker
6. Install CUDA: https://documentation.ubuntu.com/wsl/en/latest/tutorials/gpu-cuda/
7. Clone the docker image repo: git clone https://github.com/seethroughlab/runpod-worker-comfy.git
8. cd runpod-worker-comfy
9. Build the image: sudo docker build -t seethroughlab/runpod-worker-comfy:dev-base --platform linux/amd64 .
10. sudo docker-compose up
11. sudo docker login -u "seethroughlab" -p "mypassword" docker.io
12. sudo docker push seethroughlab/runpod-worker-comfy:dev-base