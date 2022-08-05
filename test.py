import os
import cv2
import time
import argparse
import sys
from pathlib import Path
from sklearn import datasets
import torch
import model.detector
import utils.utils
import glob    
import numpy as np
from threading import Thread    
import torch.backends.cudnn as cudnn  # cuda模块

from utils.general import check_requirements, clean_str
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
num_threads = min(8, os.cpu_count())  # 定义多线程个数


class LoadImages:  # for inference
    """在detect.py中使用
    load 文件夹中的图片/视频
    定义迭代器 用于detect.py
    """
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        # glob.glab: 返回所有匹配的文件路径列表   files: 提取图片所有路径
        if '*' in p:
            # 如果p是采样正则化表达式提取图片/视频, 可以使用glob获取文件路径
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            # 如果p是一个文件夹，使用glob获取全部文件路径
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            # 如果p是文件则直接获取
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        # images: 目录下所有图片的图片名  videos: 目录下所有视频的视频名
        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        # 图片与视频数量
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride   # 最大的下采样率
        self.files = images + videos  # 整合图片和视频路径到一个列表
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv  # 是不是video
        self.mode = 'image'  # 默认是读image模式
        if any(videos):
            # 判断有没有video文件  如果包含video文件，则初始化opencv中的视频模块，cap=cv2.VideoCapture等
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        """迭代器"""
        self.count = 0
        return self

    def __next__(self):
        """与iter一起用？"""
        if self.count == self.nf:  # 数据读完了
            raise StopIteration
        path = self.files[self.count]  # 读取当前文件路径

        if self.video_flag[self.count]:  # 判断当前文件是否是视频
            # Read video
            self.mode = 'video'
            # 获取当前帧画面，ret_val为一个bool变量，直到视频读取完毕之前都为True
            ret_val, img0 = self.cap.read()
            # 如果当前视频读取结束，则读取下一个视频
            if not ret_val:
                self.count += 1
                self.cap.release()
                # self.count == self.nf表示视频已经读取完了
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1  # 当前读取视频的帧数
           # print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
           # print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        res_img = cv2.resize(img0, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) 
        img = res_img.reshape(1, cfg["height"], cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0,3, 1, 2))
        img = img.to(device).float() / 255.0
        #img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
      #  img = np.ascontiguousarray(img)
        
        # 返回路径, resize+pad的图片, 原始图片, 视频对象
        return path, img, img0, self.cap

    def new_video(self, path):
        # 记录帧数
        self.frame = 0
        # 初始化视频对象
        self.cap = cv2.VideoCapture(path)
        # 得到视频文件中的总帧数
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files

class LoadStreams:
    """
    load 文件夹中视频流
    multiple IP or RTSP cameras
    定义迭代器 用于detect.py
    """
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'  # 初始化mode为images
        self.img_size = img_size
        self.stride = stride  # 最大下采样步长

        # 如果sources为一个保存了多个视频流的文件  获取每一个视频流，保存为一个列表
        if os.path.isfile(sources):

            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            # 反之，只有一个视频流文件就直接保存
            sources = [sources]

        n = len(sources)  # 视频流个数
        # 初始化图片 fps 总帧数 线程数
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later

        # 遍历每一个视频流
        for i, s in enumerate(sources):  # index, source
            # Start thread to read frames from video stream
            # 打印当前视频index/总视频数/视频流地址
            print(f'{i + 1}/{n}: {s}... ', end='')
            s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam 本地摄像头
            # s='0'打开本地摄像头，否则打开视频流地址
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'Failed to open {s}'
            # 获取视频的宽和长
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # 获取视频的帧率
            self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
            # 帧数
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

            # 读取当前画面
            _, self.imgs[i] = cap.read()  # guarantee first frame
            # 创建多线程读取视频流，daemon表示主线程结束时子线程也结束
            self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
            self.threads[i].start()
        print('')  # newline
        # check for common shapes
        # 获取进行resize+pad之后的shape，letterbox函数默认(参数auto=True)是按照矩形推理进行填充
        s = np.stack([cv2.resize(x, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR)[0].shape  for x in self.imgs],0)
        #s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, i, cap):
        # Read stream `i` frames in daemon thread
        n, f = 0, self.frames[i]
        while cap.isOpened() and n < f:
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            # 每4帧读取一次
            if n % 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[i] = im if success else self.imgs[i] * 0
            time.sleep(1 / self.fps[i])  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img0 = self.imgs.copy()
        img= [cv2.resize(x, (cfg["width"], cfg["height"]), interpolation = cv2.INTER_LINEAR) for x in img0]
        # Stack  将读取的图片拼接到一起
        img = np.stack(img, 0)

        img = img.reshape(1, cfg["height"], cfg["width"], 3)
        img = torch.from_numpy(img.transpose(0,3, 1, 2))
        img = img.to(device).float() / 255.0
        # Convert
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
      #  img = np.ascontiguousarray(img)
        

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


if __name__ == '__main__':
    #指定训练配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='', 
                        help='Specify training profile *.data')
    parser.add_argument('--weights', type=str, default=' ', 
                        help='The path of the .pth model to be transformed')
    parser.add_argument('--img', type=str, default='', 
                        help='The path of test image')

    opt = parser.parse_args()
    cfg = utils.utils.load_datafile(opt.data)
    #assert os.path.exists(opt.weights), "请指定正确的模型路径"
    #assert os.path.exists(opt.img), "请指定正确的测试图像路径"
    webcam = opt.img.isnumeric() or opt.img.endswith('.txt') or opt.img.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    counts=0
    save_img=1
    #模型加载
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.detector.Detector(cfg["classes"], cfg["anchor_num"], True).to(device)
    model.load_state_dict(torch.load(opt.weights, map_location=device))

    #sets the module in eval node
    model.eval()
    
    #数据预处理
    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(opt.img, img_size=640, stride=32)
    else:
        dataset = LoadImages(opt.img, img_size=640, stride=32)
   

    #模型推理
    start = time.perf_counter()
    vid_path,vid_writer=None,None
    for path, img,img0 ,vid_cap in dataset:
        preds = model(img)


    #特征图后处理
        output = utils.utils.handel_preds(preds, cfg, device)
        output_boxes = utils.utils.non_max_suppression(output, conf_thres = 0.3, iou_thres = 0.4)
        
        #加载label names
        LABEL_NAMES = []
        with open(cfg["names"], 'r') as f:
            for line in f.readlines():
                LABEL_NAMES.append(line.strip())
        for i ,det in enumerate(output_boxes):
            if webcam:
                im0=img0[i].copy()
                save_path='/home/yyp/Yolo-FastestV2/'+str(Path(path[i]).name)
  
            else:
                im0=img0.copy()
                save_path = '/home/yyp/Yolo-FastestV2/'+str(Path(path).name)  # img.jpg
            h, w, _ = im0.shape
            scale_h, scale_w = h / cfg["height"], w / cfg["width"]

            #绘制预测框
            for box in output_boxes[0]:
                box = box.tolist()
            
                obj_score = box[4]
                category = LABEL_NAMES[int(box[5])]

                x1, y1 = int(box[0] * scale_w), int(box[1] * scale_h)
                x2, y2 = int(box[2] * scale_w), int(box[3] * scale_h)

                cv2.rectangle(im0, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(im0, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)	
                cv2.putText(im0, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)  
            counts+=1
            cv2.imshow(save_path,im0)
            cv2.waitKey(1)  # 1 millisecond

            if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                #save_path += '.mp4'
                                save_path = str(Path(save_path).with_suffix('.mp4'))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
