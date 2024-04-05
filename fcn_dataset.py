import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
rev_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])

def paired_crop_and_resize(image, label, size):
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.5, 1.0), ratio=(1, 1))
    image = transforms.functional.resized_crop(image, i, j, h, w, size, Image.BILINEAR)
    label = Image.fromarray(label.astype('uint8'))  # Convert label back to PIL Image for resizing
    label = transforms.functional.resized_crop(label, i, j, h, w, size, Image.NEAREST)
    label = np.array(label)  # Convert label back to numpy array
    return image, label

def paired_resize(image, label, size):
    image = transforms.functional.resize(image, size, Image.BILINEAR)
    label = Image.fromarray(label.astype('uint8'))  # Convert label back to PIL Image for resizing
    label = transforms.functional.resize(label, size, Image.NEAREST)
    label = np.array(label)  # Convert label back to numpy array
    return image, label

class CamVidDataset(Dataset):
    def __init__(self, root, images_dir, labels_dir, class_dict_path, resolution, crop=False):
        self.root = root
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.resolution = resolution
        self.crop = crop
        self.class_dict = self.parse_class_dict(os.path.join(root, class_dict_path))
        self.images = [os.path.join(root, images_dir, img) for img in sorted(os.listdir(os.path.join(root, images_dir)))]
        self.labels = [os.path.join(root, labels_dir, lbl) for lbl in sorted(os.listdir(os.path.join(root, labels_dir)))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        label = np.array(label)  # Convert PIL Image to NumPy array for class ID conversion
        label = self.rgb_to_class_id(label)
        if self.crop:
            image, label = paired_crop_and_resize(image, label, self.resolution)
        else:
            image, label = paired_resize(image, label, self.resolution)
        image = transforms.ToTensor()(image)
        image = normalize(image)
        label = torch.tensor(label).long()  # Directly convert NumPy array to tensor
        return image, label

    def parse_class_dict(self, class_dict_path):
        class_dict = {}
        with open(class_dict_path, mode='r') as infile:
            reader = csv.reader(infile)
            next(reader)  # Skip the header
            for class_id, row in enumerate(reader):  # Use enumerate to generate class_id
                if len(row) != 4:  # Now expecting 4 values: name, r, g, b
                    print(f"Skipping row: {row}. Expected 4 values, got {len(row)}")
                    continue
                name, r, g, b = row
                class_dict[class_id] = ((int(r), int(g), int(b)), name)
        return class_dict

    def rgb_to_class_id(self, label_img):
        class_id_image = np.zeros((label_img.shape[0], label_img.shape[1]), dtype=np.int32)
        for class_id, (rgb_values, _) in self.class_dict.items():
            match = (label_img == np.array(rgb_values)).all(axis=-1)
            class_id_image[match] = class_id
        return class_id_image

if __name__ == "__main__":
    images_dir = "train/"
    labels_dir = "train_labels/"
    class_dict_path = "class_dict.csv"
    resolution = (240, 240)
    camvid_dataset = CamVidDataset(root='CamVid/', images_dir=images_dir, labels_dir=labels_dir, class_dict_path=class_dict_path, resolution=resolution)
    
    # Example of loading a single sample
    #image, label = camvid_dataset[0]

    label_vis = label.cpu().numpy()  # Move tensor to CPU and then convert to numpy
    label_vis = (label_vis / label_vis.max()) * 255.0  # Normalize to [0, 255]
    label_vis = label_vis.astype(np.uint8)  # Convert to uint8
    label_vis = Image.fromarray(label_vis)  # Convert numpy array to PIL Image
    label_vis.save("label_vis.png")

    image_vis = transforms.ToPILImage()(rev_normalize(image.cpu()))  # Ensure image tensor is on CPU
    image_vis.save("image_vis.png")
