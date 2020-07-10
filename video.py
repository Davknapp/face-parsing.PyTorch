import os

import torch
import torchvision.transforms as transforms
from torchvision.io import read_video, write_video
from PIL import Image
import tqdm
from model import BiSeNet
import cv2
import numpy as np

def label_images(images: torch.tensor, cp='model_final_diss.pth'):
    """
    Labels images

    Args:
        images: a (n x H x W x C) tensor with the images

    Returns:
        a (n x H x W) numpy array with the class labels
    """

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = os.path.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))

    # need to move color channels dimension to the second dimension due to weird PIL transform issues
    images = images.transpose(1, 3).transpose(2, 3)

    transform_images = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    net.eval()

    with torch.no_grad():
        labels = []
        for image in tqdm.tqdm(images, desc="segmenting frames"):
            img = transform_images(image)
            img = torch.unsqueeze(img, 0)

            img = img.cuda()
            labels_probs = net(img)[0]

            labels.append(labels_probs.squeeze(0).detach().cpu().argmax(0))

    return torch.stack(labels, dim=0)


def render_segmented_image(im: torch.tensor, labels: torch.tensor):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    vis_im = im.numpy()

    #vis_parsing_anno = labels.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(labels.numpy().astype(np.uint8), (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.35, vis_parsing_anno_color, 0.65, 0)
    vis_im = torch.tensor(vis_im)

    return vis_im

def label_and_render_video(video_path, respth='./res/test_res', cp='model_final_diss.pth'):
    frames, audio, info = read_video(video_path, pts_unit="sec")

    labels = label_images(frames, cp)
    segmented_frames = []
    for frame_inx in tqdm.tqdm(list(range(frames.shape[0])), desc="generating segmented frames"):
        segmented_frames.append(render_segmented_image(frames[frame_inx], labels[frame_inx]))

    segmented_frames = torch.stack(segmented_frames)

    write_video(os.path.join(respth, "segment"+os.path.basename(video_path))+".mp4", segmented_frames, info["video_fps"])


def replace_background(video_path, bk_img_path, respth='./res/test_res', cp='model_final_diss.pth'):
    frames, audio, info = read_video(video_path, pts_unit="sec")
    bkg = transforms.Compose(
        [
            transforms.Resize((frames[0].shape[0], frames[0].shape[1])),
            transforms.ToTensor()
        ]
    )(Image.open(bk_img_path)).transpose(0, 2).transpose(0, 1)

    # scaling to [0,255] and casting to uint8
    bkg = (bkg*255).type(torch.uint8)

    scale_labels = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((frames.shape[1], frames.shape[2]), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    labels = label_images(frames, cp)
    new_frames = []
    for frame_inx in tqdm.tqdm(list(range(frames.shape[0])), desc="generating segmented frames with background"):
        scaled_label = scale_labels(labels[frame_inx].type(torch.uint8)).squeeze(0)
        scaled_label = torch.stack([scaled_label, scaled_label, scaled_label], dim=2)

        new_frames.append(torch.where(scaled_label > 0, frames[frame_inx], bkg))

    new_frames = torch.stack(new_frames)

    write_video(os.path.join(respth, "background"+os.path.basename(video_path))+".mp4", new_frames, info["video_fps"])


def blur_background(video_path, respth='./res/test_res', cp='model_final_diss.pth'):
    frames, audio, info = read_video(video_path, 61, 65, pts_unit="sec")

    scale_labels = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((frames.shape[1], frames.shape[2]), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])

    labels = label_images(frames, cp)
    new_frames = []
    for frame_inx in tqdm.tqdm(list(range(frames.shape[0])), desc="generating segmented frames with background"):
        scaled_label = scale_labels(labels[frame_inx].type(torch.uint8)).squeeze(0)
        scaled_label = torch.stack([scaled_label, scaled_label, scaled_label], dim=2)

        blurred_img = torch.from_numpy(cv2.blur(frames[frame_inx].numpy(), (15, 15)))

        new_frames.append(torch.where(scaled_label > 0, frames[frame_inx], blurred_img))

    new_frames = torch.stack(new_frames)

    write_video(os.path.join(respth, "blurred_background"+os.path.basename(video_path))+".mp4", new_frames, round(info["video_fps"]))



if __name__ == "__main__":
    #http://conradsanderson.id.au/vidtimit/
    path_video = "/home/tbellfelix/Downloads/head.mpg"

    #https://video.bundesregierung.de/2020/03/18/a6eqjk-20200318_a-master.mp4?download=1
    path_video3 = "/home/tbellfelix/Downloads/video.mp4"

    #https://www.flickr.com/photos/hirespic/35225820042/
    bg = "/home/tbellfelix/Downloads/background.jpg"


    #evaluate(dspth='images/', cp='79999_iter.pth')
    label_and_render_video(path_video, cp='79999_iter.pth')
    replace_background(path_video, bg, cp='79999_iter.pth')
    blur_background(path_video3, cp='79999_iter.pth')
