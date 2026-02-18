import torch
import Model
import dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from imageio import imwrite
from tqdm import tqdm

if __name__ == '__main__':
    batch_size = 1
    net = Model.Net(None, img_size=384).cuda()

    ckpt = r''
    pretrained_dict = torch.load(ckpt)
    net.load_state_dict(pretrained_dict)

    Dirs = [
        r"\TestDataset\CAMO",
        r"\TestDataset\COD10K",
        r"\TestDataset\CHAMELEON",
        r"\TestDataset\NC4K",

    ]

    result_save_root = r''

    net.eval()
    for Dir in Dirs:
        dataset_name = os.path.basename(Dir)
        save_path = os.path.join(result_save_root, dataset_name)
        os.makedirs(save_path, exist_ok=True)

        Dataset = dataset.TestDataset(Dir, 384)
        Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=batch_size * 2)

        with tqdm(Dataloader, unit="batch", desc=f"Processing {dataset_name}") as tepoch:
            for data in tepoch:
                img, label = data['img'].cuda(), data['label'].cuda()
                import os

                name = os.path.basename(data['name'][0])

                with torch.no_grad():
                    out = net(img)[2]

                B, C, H, W = label.size()
                o = F.interpolate(out, (H, W), mode='bilinear', align_corners=True).detach().cpu().numpy()[0, 0]
                o = (o - o.min()) / (o.max() - o.min() + 1e-8)
                o = (o * 255).astype(np.uint8)

                imwrite(os.path.join(save_path, name), o)

    print("Test finished!")