import math
import pdb
import numpy as np
import scipy.stats
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from utils.custom_collate import SegCollate
from mms_dataloader import get_meta_split_data_loaders

class MaskGenerator (object):
    """
    Mask Generator
    """
    def generate_params(self, n_masks, mask_shape, rng=None):
        raise NotImplementedError('Abstract')

    def append_to_batch(self, *batch):
        x = batch[0]
        params = self.generate_params(len(x), x.shape[2:4])
        return batch + (params,)

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        raise NotImplementedError('Abstract')


class BoxMaskGenerator (MaskGenerator):
    def __init__(self, prop_range, n_boxes=1, random_aspect_ratio=True, prop_by_area=True, within_bounds=True, invert=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # Choose the proportion of each mask that should be above the threshold
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        return t_params


class AddMaskParamsToBatch (object):
    """
    We add the cut-and-paste parameters to the mini-batch within the collate function,
    (we pass it as the `batch_aug_fn` parameter to the `SegCollate` constructor)
    as the collate function pads all samples to a common size
    """
    def __init__(self, mask_gen):
        self.mask_gen = mask_gen

    def __call__(self, batch):
        sample = batch[0]
        mask_size = sample['img'].shape[1:3]
        # print("The size is ", mask_size)
        params = self.mask_gen.generate_params(len(batch), mask_size)
        for sample, p in zip(batch, params):
            sample['mask_params'] = p.astype(np.float32)
        return batch


if __name__ == '__main__':
    cutmix_mask_prop_range = (0.25, 0.5)
    cutmix_boxmask_n_boxes = 1
    cutmix_boxmask_fixed_aspect_ratio = False
    cutmix_boxmask_by_size = False
    cutmix_boxmask_outside_bounds = False
    cutmix_boxmask_no_invert = False
    mask_generator = BoxMaskGenerator(prop_range= cutmix_mask_prop_range, n_boxes= cutmix_boxmask_n_boxes,
                                           random_aspect_ratio=not  cutmix_boxmask_fixed_aspect_ratio,
                                           prop_by_area=not  cutmix_boxmask_by_size, within_bounds=not  cutmix_boxmask_outside_bounds,
                                           invert=not  cutmix_boxmask_no_invert)
    add_mask_params_to_batch = AddMaskParamsToBatch(
        mask_generator
    )

    collate_fn = SegCollate()
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset, \
        domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
        test_dataset = get_meta_split_data_loaders(
            test_vendor='A', image_size=224)

    label_loader = DataLoader(dataset=domain_1_labeled_dataset, batch_size=32, num_workers=4,
                              shuffle=True, drop_last=True, collate_fn=mask_collate_fn)

    label_dataloader = iter(label_loader)
    output = label_dataloader.next()
    img = output['img']
    mask = output['mask']
    domain_label = output['domain_label']
    mask_params = output['mask_params']
    print(output.keys())
    print(img.shape)
    print(mask.shape)
    print(domain_label.shape)
    print(mask_params.shape)



