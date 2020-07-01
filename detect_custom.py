# %%
import cv2

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *


def __detect(image, opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt['output'], opt['source'], opt['weights'], opt['view_img'], opt['save_txt'], opt['img_size']
    webcam = False

    # Initialize
    device = torch_utils.select_device(opt['device'])
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    # half = device.type != 'cpu'  # half precision only supported on CUDA
    half = False

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    imgsz = check_img_size(imgsz, s=model.model[-1].stride.max())  # check img_size

    # Set Dataloader
    dataset = LoadInMemoryImage(image, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    res = []
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment = opt['augment'])[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt['conf_thres'], opt['iou_thres'], classes=opt['classes'], agnostic=opt['agnostic_nms'])
        t2 = torch_utils.time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            im0 = im0s
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                detections = []
                for *xyxy, conf, cls in det:
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    detections.append({
                        'class': cls.item(),
                        'x': xywh[0],
                        'y': xywh[1],
                        'w': xywh[2],
                        'h': xywh[3],
                        'confidence': conf
                    })
                res.append(detections)
    return res

# %%

# %%
def detect():
    impath = r'C:\Users\adamw\OneDrive\Pulpit\Cephio\yolov3\data\images\0001.jpg'
    image = cv2.imread(impath)
    opt = dict(
        weights = 'weights/best.pt',
        source = 'data/images',
        output = 'inference/output',
        img_size = 512,
        conf_thres = 0.2,
        iou_thres = 0.6,
        fourcc = 'mp4v',
        device = 'cpu',
        view_img = False,
        save_txt = True,
        classes = False,
        agnostic_nms = False,
        augment = False,
    )

    with torch.no_grad():
        res = __detect(image, opt)
        print(res)

        # Update all models
        # for opt['weights'] in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt', 'yolov3-spp.pt']:
        #    detect()
        #    create_pretrained(opt['weights'], opt['weights'])

# %%
