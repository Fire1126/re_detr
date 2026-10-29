# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import List, Optional, Union
import torch
import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform, to_tensor
from mmcv.transforms import CenterCrop as MMCV_CenterCrop
from mmcv.transforms.utils import cache_randomness
from mmdet.structures.bbox import BaseBoxes, get_box_type
from mmdet.structures.mask import PolygonMasks
from mmengine.utils import is_list_of

from mmrotate.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ConvertBoxType(BaseTransform):
    """Convert boxes in results to a certain box type.

    Args:
        box_type_mapping (dict): A dictionary whose key will be used to search
            the item in `results`, the value is the destination box type.
    """

    def __init__(self, box_type_mapping: dict) -> None:
        self.box_type_mapping = box_type_mapping

    def transform(self, results: dict) -> dict:
        """The transform function."""
        for key, dst_box_type in self.box_type_mapping.items():
            if key not in results:
                continue
            assert isinstance(results[key], BaseBoxes), \
                f"results['{key}'] not a instance of BaseBoxes."
            results[key] = results[key].convert_to(dst_box_type)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(box_type_mapping={self.box_type_mapping})'
        return repr_str


@TRANSFORMS.register_module()
class Rotate(BaseTransform):
    """Rotate the images, bboxes, masks and segmentation map by a certain
    angle. Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)
    Modified Keys:
    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    Added Keys:
    - homography_matrix
    Args:
        rotate_angle (int): An angle to rotate the image.
        img_border_value (int or float or tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 0.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 rotate_angle: int,
                 img_border_value: Union[int, float, tuple] = 0,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        if isinstance(img_border_value, (float, int)):
            img_border_value = tuple([float(img_border_value)] * 3)
        elif isinstance(img_border_value, tuple):
            assert len(img_border_value) == 3, \
                f'img_border_value as tuple must have 3 elements, ' \
                f'got {len(img_border_value)}.'
            img_border_value = tuple([float(val) for val in img_border_value])
        else:
            raise ValueError(
                'img_border_value must be float or tuple with 3 elements.')
        self.rotate_angle = rotate_angle
        self.img_border_value = img_border_value
        self.mask_border_value = mask_border_value
        self.seg_ignore_label = seg_ignore_label
        self.interpolation = interpolation

    def _get_homography_matrix(self, results: dict) -> np.ndarray:
        """Get the homography matrix for Rotate."""
        img_shape = results['img_shape']
        center = ((img_shape[1] - 1) * 0.5, (img_shape[0] - 1) * 0.5)
        cv2_rotation_matrix = cv2.getRotationMatrix2D(center,
                                                      -self.rotate_angle, 1.0)
        return np.concatenate(
            [cv2_rotation_matrix,
             np.array([0, 0, 1]).reshape((1, 3))],
            dtype=np.float32)

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the geometric transformation."""
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = self.homography_matrix
        else:
            results['homography_matrix'] = self.homography_matrix @ results[
                'homography_matrix']

    def _transform_img(self, results: dict) -> None:
        """Rotate the image."""
        results['img'] = mmcv.imrotate(
            results['img'],
            self.rotate_angle,
            border_value=self.img_border_value,
            interpolation=self.interpolation)

    def _transform_masks(self, results: dict) -> None:
        """Rotate the masks."""
        results['gt_masks'] = results['gt_masks'].rotate(
            results['img_shape'],
            self.rotate_angle,
            border_value=self.mask_border_value,
            interpolation=self.interpolation)

    def _transform_seg(self, results: dict) -> None:
        """Rotate the segmentation map."""
        results['gt_seg_map'] = mmcv.imrotate(
            results['gt_seg_map'],
            self.rotate_angle,
            border_value=self.seg_ignore_label,
            interpolation='nearest')

    def _transform_bboxes(self, results: dict) -> None:
        """Rotate the bboxes."""
        if len(results['gt_bboxes']) == 0:
            return
        img_shape = results['img_shape']
        center = (img_shape[1] * 0.5, img_shape[0] * 0.5)
        results['gt_bboxes'].rotate_(center, self.rotate_angle)
        results['gt_bboxes'].clip_(img_shape)

    def _filter_invalid(self, results: dict) -> None:
        """Filter invalid data w.r.t `gt_bboxes`"""
        # results['img_shape'] maybe (h,w,c) or (h,w)
        height, width = results['img_shape'][:2]
        if 'gt_bboxes' in results:
            if len(results['gt_bboxes']) == 0:
                return
            bboxes = results['gt_bboxes']
            valid_index = results['gt_bboxes'].is_inside([height,
                                                          width]).numpy()
            results['gt_bboxes'] = bboxes[valid_index]

            # ignore_flags
            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_index]

            # labels
            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                    valid_index]

            # mask fields
            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_index.nonzero()[0]]

    def transform(self, results: dict) -> dict:
        """The transform function."""
        self.homography_matrix = self._get_homography_matrix(results)
        self._record_homography_matrix(results)
        self._transform_img(results)
        if results.get('gt_bboxes', None) is not None:
            self._transform_bboxes(results)
        if results.get('gt_masks', None) is not None:
            self._transform_masks(results)
        if results.get('gt_seg_map', None) is not None:
            self._transform_seg(results)
        self._filter_invalid(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_angle={self.rotate_angle}, '
        repr_str += f'img_border_value={self.img_border_value}, '
        repr_str += f'mask_border_value={self.mask_border_value}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Random rotate image & bbox & masks. The rotation angle will choice in.

    [-angle_range, angle_range). Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)
    Modified Keys:
    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    Added Keys:
    - homography_matrix
    Args:
        prob (float): The probability of whether to rotate or not. Defaults
            to 0.5.
        angle_range (int): The maximum range of rotation angle. The rotation
            angle will lie in [-angle_range, angle_range). Defaults to 180.
        rect_obj_labels (List[int], Optional): A list of labels whose
            corresponding objects are alwags horizontal. If
            results['gt_bboxes_labels'] has any label in ``rect_obj_labels``,
            the rotation angle will only be choiced from [90, 180, -90, -180].
            Defaults to None.
        rotate_type (str): The type of rotate class to use. Defaults to
            "Rotate".
        **rotate_kwargs: Other keyword arguments for the ``rotate_type``.
    """

    def __init__(self,
                 prob: float = 0.5,
                 angle_range: int = 180,
                 rect_obj_labels: Optional[List[int]] = None,
                 rotate_type: str = 'Rotate',
                 **rotate_kwargs) -> None:
        assert 0 < angle_range <= 180
        self.prob = prob
        self.angle_range = angle_range
        self.rect_obj_labels = rect_obj_labels
        self.rotate_cfg = dict(type=rotate_type, **rotate_kwargs)
        self.rotate = TRANSFORMS.build({'rotate_angle': 0, **self.rotate_cfg})
        self.horizontal_angles = [90, 180, -90, -180]

    @cache_randomness
    def _random_angle(self) -> int:
        """Random angle."""
        return self.angle_range * (2 * np.random.rand() - 1)

    @cache_randomness
    def _random_horizontal_angle(self) -> int:
        """Random horizontal angle."""
        return np.random.choice(self.horizontal_angles)

    @cache_randomness
    def _is_rotate(self) -> bool:
        """Randomly decide whether to rotate."""
        return np.random.rand() < self.prob

    def transform(self, results: dict) -> dict:
        """The transform function."""
        if not self._is_rotate():
            return results

        rotate_angle = self._random_angle()
        if self.rect_obj_labels is not None and 'gt_bboxes_labels' in results:
            for label in self.rect_obj_labels:
                if (results['gt_bboxes_labels'] == label).any():
                    rotate_angle = self._random_horizontal_angle()
                    break

        self.rotate.rotate_angle = rotate_angle
        return self.rotate(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'rotate_angle={self.angle_range}, '
        repr_str += f'rect_obj_labels={self.rect_obj_labels}, '
        repr_str += f'rotate_cfg={self.rotate_cfg})'
        return repr_str


@TRANSFORMS.register_module()
class RandomChoiceRotate(BaseTransform):
    """Random rotate image & bbox & masks from a list of angles. Rotation angle
    will be randomly choiced from ``angles``. Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)
    Modified Keys:
    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    Added Keys:
    - homography_matrix
    Args:
        angles (list[int]): Angles for rotation. 0 is the default value for
            non-rotation and shouldn't be included in ``angles``.
        prob (float or list[float]): If ``prob`` is a float, it is the
            probability of whether to rotate. If ``prob`` is a list, it is
            the probabilities of each rotation angle in ``angles``.
        rect_obj_labels (List[int]): A list of labels whose corresponding
            objects are alwags horizontal. If results['gt_bboxes_labels'] has
            any label in ``rect_obj_labels``, the rotation angle will only be
            choiced from [90, 180, -90, -180].
        rotate_type (str): The type of rotate class to use. Defaults to
            "Rotate".
        **rotate_kwargs: Other keyword arguments for the ``rotate_type``.
    """

    def __init__(self,
                 angles,
                 prob: Union[float, List[float]] = 0.5,
                 rect_obj_labels=None,
                 rotate_type='Rotate',
                 **rotate_kwargs) -> None:
        if isinstance(prob, list):
            assert is_list_of(prob, Number)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, Number):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be number or list of number, but \
                              got `{type(prob)}`.')
        self.prob = prob

        assert isinstance(angles, list) and is_list_of(angles, int)
        assert 0 not in angles
        self.angles = angles
        if isinstance(self.prob, list):
            assert len(self.prob) == len(self.angles)

        self.rect_obj_labels = rect_obj_labels
        self.rotate_cfg = dict(type=rotate_type, **rotate_kwargs)
        self.rotate = TRANSFORMS.build({'rotate_angle': 0, **self.rotate_cfg})
        self.horizontal_angles = [90, 180, -90, -180]

    @cache_randomness
    def _choice_angle(self) -> int:
        """Choose the angle."""
        angle_list = self.angles + [0]
        if isinstance(self.prob, list):
            non_prob = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        else:
            non_prob = 1. - self.prob
            single_ratio = self.prob / (len(angle_list) - 1)
            prob_list = [single_ratio] * (len(angle_list) - 1) + [non_prob]
        angle = np.random.choice(angle_list, p=prob_list)
        return angle

    @cache_randomness
    def _random_horizontal_angle(self) -> int:
        """Random horizontal angle."""
        return np.random.choice(self.horizontal_angles)

    def transform(self, results: dict) -> dict:
        """The transform function."""
        rotate_angle = self._choice_angle()
        if rotate_angle == 0:
            return results

        if self.rect_obj_labels is not None and 'gt_bboxes_labels' in results:
            for label in self.rect_obj_labels:
                if (results['gt_bboxes_labels'] == label).any():
                    rotate_angle = self._random_horizontal_angle()
                    break

        self.rotate.rotate_angle = rotate_angle
        return self.rotate(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(angles={self.angles}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'rect_obj_labels={self.rect_obj_labels}, '
        repr_str += f'rotate_cfg={self.rotate_cfg})'
        return repr_str


@TRANSFORMS.register_module()
class ConvertMask2BoxType(BaseTransform):
    """Convert masks in results to a certain box type.

    Required Keys:

    - ori_shape
    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_masks (BitmapMasks | PolygonMasks)
    - instances (List[dict]) (optional)
    Modified Keys:
    - gt_bboxes
    - gt_masks
    - instances

    Args:
        box_type (str): The destination box type.
        keep_mask (bool): Whether to keep the ``gt_masks``.
            Defaults to False.
    """

    def __init__(self, box_type: str, keep_mask: bool = False) -> None:
        _, self.box_type_cls = get_box_type(box_type)
        assert hasattr(self.box_type_cls, 'from_instance_masks')
        self.keep_mask = keep_mask

    def transform(self, results: dict) -> dict:
        """The transform function."""
        assert 'gt_masks' in results.keys()
        masks = results['gt_masks']
        results['gt_bboxes'] = self.box_type_cls.from_instance_masks(masks)
        if not self.keep_mask:
            results.pop('gt_masks')

        # Modify results['instances'] for RotatedCocoMetric
        if results.get('instances', []):
            converted_instances = []
            for instance in results['instances']:
                m = np.array(instance['mask'][0])
                m = PolygonMasks([[m]], results['ori_shape'][1],
                                 results['ori_shape'][0])
                instance['bbox'] = self.box_type_cls.from_instance_masks(
                    m).tensor[0].numpy().tolist()

                if not self.keep_mask:
                    instance.pop('mask')
                converted_instances.append(instance)
            results['instances'] = converted_instances

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(box_type_cls={self.box_type_cls}, '
        repr_str += f'keep_mask={self.keep_mask})'
        return repr_str

@TRANSFORMS.register_module()
class ConvertWeakSupervision(BaseTransform):
    """Convert RBoxes to Single Center Points."""

    def __init__(self,
                 point_proportion: float = 0.3,
                 hbox_proportion: float = 0.3,
                 point_dummy: float = 1,
                 hbox_dummy: float = 0,
                 modify_labels: bool = False) -> None:
        self.point_proportion = point_proportion
        self.hbox_proportion = hbox_proportion
        self.point_dummy = point_dummy
        self.hbox_dummy = hbox_dummy
        self.modify_labels = modify_labels

    def transform(self, results: dict) -> dict:
        """The transform function."""

        max_idx_p = int(round(results['gt_bboxes'].tensor.shape[0] * self.point_proportion))
        results['gt_bboxes'].tensor[:max_idx_p, 2] = self.point_dummy
        results['gt_bboxes'].tensor[:max_idx_p, 3] = self.point_dummy
        results['gt_bboxes'].tensor[:max_idx_p, 4] = 0

        max_idx_h = max_idx_p + int(round(results['gt_bboxes'].tensor.shape[0] * self.hbox_proportion))
        results['gt_bboxes'][max_idx_p:max_idx_h] = \
            results['gt_bboxes'][max_idx_p:max_idx_h].convert_to('hbox').convert_to('rbox')
        results['gt_bboxes'].tensor[max_idx_p:max_idx_h, 4] = self.hbox_dummy

        if self.modify_labels:
            ws_types = torch.zeros(results['gt_bboxes'].tensor.shape[0], dtype=torch.long)
            ws_types[:max_idx_p] = 2
            ws_types[max_idx_p:max_idx_h] = 1
            labels = to_tensor(results['gt_bboxes_labels'])
            results['gt_bboxes_labels'] = torch.stack((labels, ws_types), -1)

        return results

@TRANSFORMS.register_module()
class RegularizeRotatedBox(BaseTransform):
    """Regularize rotated boxes.

    Due to the angle periodicity, one rotated box can be represented in
    many different (x, y, w, h, t). To make each rotated box unique,
    ``regularize_boxes`` will take the remainder of the angle divided by
    180 degrees.

    For convenience, three angle_version can be used here:

    - 'oc': OpenCV Definition. Has the same box representation as
        ``cv2.minAreaRect`` the angle ranges in [-90, 0).
    - 'le90': Long Edge Definition (90). the angle ranges in [-90, 90).
        The width is always longer than the height.
    - 'le135': Long Edge Definition (135). the angle ranges in [-45, 135).
        The width is always longer than the height.

    Required Keys:

    - gt_bboxes (RotatedBoxes[torch.float32])

    Modified Keys:

    - gt_bboxes

    Args:
        angle_version (str): Angle version. Can only be 'oc',
            'le90', or 'le135'. Defaults to 'le90.
    """

    def __init__(self, angle_version='le90') -> None:
        self.angle_version = angle_version
        from mmrotate.structures.bbox import RotatedBoxes
        self.box_type = RotatedBoxes

    def transform(self, results: dict) -> dict:
        assert isinstance(results['gt_bboxes'], self.box_type)
        results['gt_bboxes'] = self.box_type(
            results['gt_bboxes'].regularize_boxes(self.angle_version))
        return results

@TRANSFORMS.register_module()
class CenterCrop(MMCV_CenterCrop):

    def _crop_bboxes(self, results: dict, bboxes: np.ndarray) -> None:
        """Update bounding boxes according to CenterCrop.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if 'gt_bboxes' in results:
            offset_w = bboxes[0]
            offset_h = bboxes[1]
            gt_bboxes = results['gt_bboxes']
            last_dim = gt_bboxes.shape[-1]
            img_h, img_w = results['img'].shape[:2]

            if last_dim == 4:
                # (tl_x, tl_y, br_x, br_y)
                bbox_offset = np.array([offset_w, offset_h,offset_w, offset_h],
                                       dtype=gt_bboxes.dtype)
                gt_bboxes -= bbox_offset

                if self.clip_object_border:
                    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_w)
                    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_h)

            elif last_dim == 5:
                # (cx, cy, w, h, angle)
                gt_bboxes.tensor[:, 0] -= offset_w
                gt_bboxes.tensor[:, 1] -= offset_h

                if self.clip_object_border:
                    raise NotImplementedError

            elif last_dim == 8:
                # (x1, y1, x2, y2, x3, y3, x4, y4)
                bbox_offset = torch.tensor([offset_w, offset_h] * 4, dtype=gt_bboxes.dtype)
                gt_bboxes.tensor -= bbox_offset

                if self.clip_object_border:
                    raise NotImplementedError


@TRANSFORMS.register_module()
class RandomTranslate(BaseTransform):
    """Randomly translate the image and the bounding boxes.

    Compared with :obj:`mmdet.RandomShift`, this implementation is friendly
    to rotated boxes (:obj:`RotatedBoxes` / :obj:`QuadriBoxes`, i.e.
    :obj:`BaseBoxes`):

    1. The border exposed by the translation is filled with ``pad_val``
       instead of ``0``, so it is consistent with :obj:`mmdet.Pad`.
    2. The boxes whose center is translated out of the image are filtered
       out. Clipping a rotated box is ill-defined (``RotatedBoxes.clip_``
       does nothing), so keeping a fully out-of-image rotated box as a valid
       label would inject noise into training.

    Required Keys:

    - img (np.ndarray)
    - gt_bboxes (BaseBoxes)
    - gt_bboxes_labels (np.int64)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_bboxes_labels
    - gt_ignore_flags (optional)

    Args:
        prob (float): Probability of translating. Defaults to 0.5.
        max_shift_px (int): The max pixels for shifting along each axis.
            Defaults to 100.
        pad_val (int or Sequence[int]): The value used to fill the exposed
            border. Defaults to 114 (consistent with :obj:`mmdet.Pad`).
        filter_thr_px (int): The width / height threshold for filtering
            degenerate boxes. Defaults to 1.
    """

    def __init__(self,
                 prob: float = 0.5,
                 max_shift_px: int = 100,
                 pad_val=114,
                 filter_thr_px: int = 1) -> None:
        assert 0 <= prob <= 1, f'prob must be in [0, 1], but got {prob}'
        assert max_shift_px >= 0
        self.prob = prob
        self.max_shift_px = max_shift_px
        self.pad_val = pad_val
        self.filter_thr_px = int(filter_thr_px)

    def transform(self, results: dict) -> dict:
        """Transform function to randomly translate the image and boxes.

        Args:
            results (dict): Result dict from the loading pipeline.

        Returns:
            dict: Translated results.
        """
        if np.random.uniform(0, 1) >= self.prob:
            return results

        img = results['img']
        img_h, img_w = img.shape[:2]

        shift_x = np.random.randint(-self.max_shift_px, self.max_shift_px + 1)
        shift_y = np.random.randint(-self.max_shift_px, self.max_shift_px + 1)

        gt_bboxes = results['gt_bboxes']
        assert isinstance(gt_bboxes, BaseBoxes), \
            '`gt_bboxes` should be an instance of `BaseBoxes`, but got ' \
            f'{type(gt_bboxes)}. Please make sure `ConvertBoxType` is ' \
            'placed before `RandomTranslate` in the pipeline.'

        bboxes = gt_bboxes.clone()
        bboxes.translate_([shift_x, shift_y])

        # Filter out the boxes whose center is translated out of the image
        # and the degenerate ones.
        centers = bboxes.tensor[:, :2]
        valid_inds = ((centers[:, 0] >= 0) & (centers[:, 0] < img_w) &
                      (centers[:, 1] >= 0) & (centers[:, 1] < img_h) &
                      (bboxes.widths > self.filter_thr_px) &
                      (bboxes.heights > self.filter_thr_px)).numpy()

        # If every box is shifted out of the image, skip this sample.
        if not valid_inds.any():
            return results

        results['gt_bboxes'] = bboxes[valid_inds]
        results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_inds]
        if results.get('gt_ignore_flags', None) is not None:
            results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_inds]

        # Translate the image, filling the exposed border with `pad_val`.
        new_x, ori_x = max(0, shift_x), max(0, -shift_x)
        new_y, ori_y = max(0, shift_y), max(0, -shift_y)
        new_w = img_w - abs(shift_x)
        new_h = img_h - abs(shift_y)

        if isinstance(self.pad_val, (list, tuple)):
            new_img = np.empty_like(img)
            new_img[...] = np.array(self.pad_val, dtype=img.dtype)
        else:
            new_img = np.full_like(img, self.pad_val)
        new_img[new_y:new_y + new_h, new_x:new_x + new_w] = \
            img[ori_y:ori_y + new_h, ori_x:ori_x + new_w]
        results['img'] = new_img

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'max_shift_px={self.max_shift_px}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'filter_thr_px={self.filter_thr_px})'
        return repr_str


@TRANSFORMS.register_module()
class RandomGaussianNoise(BaseTransform):
    """Add Gaussian noise to the image.

    The noise is added in the value range of the input image (e.g.
    ``[0, 255]`` for an ``np.uint8`` image), and the noised image is clipped
    back to the valid range of its dtype when ``clip=True`` (only for integer
    images; for float images a ``clip_range`` must be given explicitly).

    Note: :obj:`mmdet.PhotoMetricDistortion` converts the image to
    ``float32`` in ``[0, 255]`` and does **not** cast it back to ``uint8``.
    To keep ``sigma`` in an intuitive ``[0, 255]`` scale, place this transform
    **before** :obj:`mmdet.PhotoMetricDistortion`.

    Required Keys:

    - img (np.ndarray)

    Modified Keys:

    - img

    Args:
        prob (float): Probability of adding noise. Defaults to 0.5.
        mean (float or tuple): Mean of the Gaussian noise. If a tuple, a
            value is uniformly sampled from ``(mean[0], mean[1])`` for each
            image. Defaults to 0.0.
        sigma (float or tuple): Standard deviation of the Gaussian noise. If
            a tuple, a value is uniformly sampled from ``(sigma[0], sigma[1])``
            for each image. Defaults to (5.0, 15.0).
        per_channel (bool): Whether to sample the noise independently for
            each channel. Defaults to False.
        clip (bool): Whether to clip the noised image back to a valid range.
            Defaults to True.
        clip_range (tuple, optional): The ``(min, max)`` range used to clip
            the noised image. Only used when ``clip=True`` and the image is
            not an integer array. Defaults to None.
    """

    def __init__(self,
                 prob: float = 0.5,
                 mean=0.0,
                 sigma=(5.0, 15.0),
                 per_channel: bool = False,
                 clip: bool = True,
                 clip_range: Optional[tuple] = None) -> None:
        assert 0 <= prob <= 1, f'prob must be in [0, 1], but got {prob}'
        self.prob = prob
        self.mean = mean
        self.sigma = sigma
        self.per_channel = per_channel
        self.clip = clip
        self.clip_range = clip_range

    @staticmethod
    def _sample(value) -> float:
        if isinstance(value, (tuple, list)):
            return float(np.random.uniform(value[0], value[1]))
        return float(value)

    def transform(self, results: dict) -> dict:
        """Transform function to add Gaussian noise to the image.

        Args:
            results (dict): Result dict from the loading pipeline.

        Returns:
            dict: Result dict with the noised image.
        """
        if np.random.uniform(0, 1) >= self.prob:
            return results

        img = results['img']
        ori_dtype = img.dtype
        mean = self._sample(self.mean)
        sigma = self._sample(self.sigma)

        # `np.random.normal` requires sigma > 0.
        if sigma <= 0:
            return results

        # Add the noise in-place in float32 to avoid a large float64 temp.
        img = img.astype(np.float32)
        if self.per_channel:
            noise = np.random.normal(mean, sigma, size=img.shape)
            img += noise
        else:
            noise = np.random.normal(mean, sigma, size=img.shape[:2])
            if img.ndim == 3:
                img += noise[..., None]
            else:
                img += noise

        if self.clip:
            if np.issubdtype(ori_dtype, np.integer):
                info = np.iinfo(ori_dtype)
                img = np.clip(img, info.min, info.max)
            elif self.clip_range is not None:
                img = np.clip(img, self.clip_range[0], self.clip_range[1])

        results['img'] = img.astype(ori_dtype)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'mean={self.mean}, '
        repr_str += f'sigma={self.sigma}, '
        repr_str += f'per_channel={self.per_channel}, '
        repr_str += f'clip={self.clip}, '
        repr_str += f'clip_range={self.clip_range})'
        return repr_str