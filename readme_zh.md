# Propainter 的一个简单 web-ui
一个可以快速选择物体并将其从视频中消除的web-ui

## 效果演示
![](./demo.gif)

## 使用
如果不想安装环境也可以直接下载整合包，下载后双击start.bat即可\
下载链接 [百度网盘](https://pan.baidu.com/s/1XkQhzCzTtzVfgQg5heQQrA?pwd=jo38 )\
使用教程 [bilibili](https://www.bilibili.com/video/BV1NH4y1o7mS/) [youtube](https://www.youtube.com/watch?v=CcivHjbHIcQ)

### 克隆项目到本地
```
git clone https://github.com/halfzm/ProPainiter-Webui.git
```

### 创建虚拟环境
```
conda create -n propainter python=3.10
conda activate propainter
```

### 安装依赖
请参考 [Segment-ant-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
 和 [ProPainter](https://github.com/sczhou/ProPainter) 项目中的安装要求（P.S.无需安装groundingdino）
```
pip install -r requirements.txt
```

### 下载预训练模型
下载 propainter 需要的模型 \
[propainter](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth)\
[raft-things](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth)\
[recurrent_flow_completion](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth)\
[i3d_rgb_imagenet](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/i3d_rgb_imagenet.pt)

下载 segment-and-track-anything 所需的模型\
SAM-VIT-B ([sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth))\
R50-DeAOT-L ([R50_DeAOTL_PRE_YTB_DAV.pth](https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view))\
GroundingDINO-T ([groundingdino_swint_ogc](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth))\

下载后的文件结构应该如下：
```
ckpt
   |- bert-base-uncased
   |- groundingdino_swint_ogc.pth
   |- R50_EdAOTL_PRE_YTB_DAV.pth
   |- sam_vit_b_01ec64.pth
...
ProPainter/weights
   |- ProPainter.pth
   |- recurrent_flow_completion.pth
   |- raft-things.pth
   |- i3d_rgb_imagenet.pt (for evaluating VFID metric)
   |- README.md
```

### 快速启动
```
python app.py
```


## 参考
 - [Segment-ant-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
 - [ProPainter](https://github.com/sczhou/ProPainter)


## 星标历史
[![Star History Chart](https://api.star-history.com/svg?repos=halfzm/ProPainter-Webui&type=Date)](https://star-history.com/#halfzm/ProPainter-Webui&Date)