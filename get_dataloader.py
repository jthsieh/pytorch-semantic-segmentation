import torch.utils.data as data
import torchvision.transforms as standard_transforms

from data_pipeline.nyu_d2 import *
import segmentation.utils.transforms as extended_transforms

ALL_MODALITIES = ['rgb', 'depth']

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def get_dataloader(opt):

  input_transform = standard_transforms.Compose([
      standard_transforms.ToTensor(),
      standard_transforms.Normalize(*mean_std)
  ])
  target_transform = extended_transforms.MaskToTensor()

  dset = NYU_D2(opt.root, opt.modality, opt.train, transform=input_transform,
                target_transform=target_transform)

  idx_t = 0 if opt.train else 1
  dataloader = data.DataLoader(dset, batch_size=opt.batch_sizes[idx_t], shuffle=opt.train,
                               num_workers=opt.n_workers)


  return dataloader
