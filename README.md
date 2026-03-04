# DCP 暗通道先验去雾（Jetson / Docker）  

本项目实现了基于 **暗通道先验（DCP）+ Guided Filter** 的单图去雾算法，并提供 Docker 镜像，便于在 **NVIDIA Jetson（ARM64/aarch64）** 上直接运行，无需手动配置 OpenCV/NumPy 环境。  

- GitHub 仓库：https://github.com/FireFly0922/dcp-dehaze.git  
- Docker Hub 镜像：`firefly0922/dcp-dehaze`  
- 示例输入图：`薄雾图.png`  

## 适用环境  

- **Jetson（ARM64/aarch64）** 设备（如 Xavier / Orin 系列）  
- 推荐 JetPack 5.x（Ubuntu 20.04）  
- 已安装 Docker（请保持它运行，且使用的网络环境能访问Docker Hub）  
> 注意：该镜像为 **ARM64** 架构，不适用于 x86_64 的普通 PC 直接运行。  

## 快速开始  

### 1) 拉取项目代码（获取示例图片/脚本/说明）  

'''bash
git clone https://github.com/FireFly0922/dcp-dehaze.git
cd dcp-dehaze  

### 2）拉取Docker镜像  

'''bash
sudo docker pull firefly0922/dcp-dehaze:latest  

### 3）运行去雾  

'''bash
sudo docker run --rm -it \
  -v "$PWD":/data \
  firefly0922/dcp-dehaze:r35.5.0-v1 \
  "/data/薄雾图.png" "/data/out.png"  

运行成功后，你会在当前目录得到 out.png，终端会输出类似：  
去雾完成，已保存到：/data/out.png  

## 参数说明  

脚本支持调整以下参数：  
--win：暗通道窗口大小（默认 15）  
--omega：去雾强度系数（默认 0.95）  
--t0：传输图下限（默认 0.1）  
--gf_r：导向滤波半径（默认 40）  
--gf_eps：导向滤波 eps（默认 1e-3）  

调参方法（例）：  
'''bash
sudo docker run --rm -it \
  -v "$PWD":/data \
  firefly0922/dcp-dehaze:r35.5.0-v1 \
  "/data/薄雾图.png" "/data/out.png" \
  --win 15 --omega 0.95 --t0 0.1 --gf_r 40 --gf_eps 1e-3
