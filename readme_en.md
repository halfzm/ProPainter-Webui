# A webui for Propainter
A webui that you can easily pick up objects from the video and eliminate them.

## Demo
![](./demo.gif)

## Getting started
If you don't want to install the environment, you can download the package directly.\
link [百度网盘](https://pan.baidu.com/s/1XkQhzCzTtzVfgQg5heQQrA?pwd=jo38 )\
tutorial [bilibili](https://www.bilibili.com/video/BV1NH4y1o7mS/) [youtube](https://www.youtube.com/watch?v=CcivHjbHIcQ)

### clone repo
```
git clone https://github.com/halfzm/ProPainiter-Webui.git
```

### create conda enviroment
```
conda create -n propainter python=3.10
conda activate propainter
```

### install dependencies
Just follow the instructions in [Segment-ant-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
 and [ProPainter](https://github.com/sczhou/ProPainter)（P.S.Don't need to install groundingdino, I have put it in the project.）
```
pip install -r requirements.txt
```

### prepare pretrained models
Download all the needed models for propainter \
[propainter](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth)\
[raft-things](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth)\
[recurrent_flow_completion](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth)\
[i3d_rgb_imagenet](https://github.com/sczhou/ProPainter/releases/download/v0.1.0/i3d_rgb_imagenet.pt)

Download all the needed models for segment-and-track-anything\
SAM-VIT-B ([sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth))\
R50-DeAOT-L ([R50_DeAOTL_PRE_YTB_DAV.pth](https://drive.google.com/file/d/1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ/view))\
GroundingDINO-T ([groundingdino_swint_ogc](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth))\

The directory structure will be arranged as：
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

### quick start
```
python app.py
```


## Reference
 - [Segment-ant-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
 - [ProPainter](https://github.com/sczhou/ProPainter)


## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=halfzm/ProPainter-Webui&type=Date)](https://star-history.com/#halfzm/ProPainter-Webui&Date)
