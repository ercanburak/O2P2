from torch.utils.data import Dataset
from PIL import Image


class O2P2Dataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        img0_path, img1_path, seg_paths = self.dataset[index]
        img0 = Image.open(img0_path).convert('RGB')
        img1 = Image.open(img1_path).convert('RGB')
        segs = []
        for seg_path in seg_paths:
            seg = Image.open(seg_path).convert('RGB')
            segs.append(seg)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            for seg in segs:
                seg = self.transform(seg)
        return img0, img1, segs

    def __len__(self):
        return len(self.dataset)