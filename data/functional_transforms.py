"""
This file should be remove at pytorch release > 7.0
It uses functions from these files:
https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
"""
import math
import warnings

from typing import Optional, Dict, Tuple

import torch

from PIL import Image
from torch import Tensor
from torch.jit.annotations import List
from torch.nn.functional import grid_sample


def _is_tensor_a_torch_image(x: Tensor) -> bool:
    """
    From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
    """
    return x.ndim >= 2


def _get_image_size(img: Tensor) -> List[int]:
    """
    From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
    Returns (w, h) of tensor image
    """
    if _is_tensor_a_torch_image(img):
        return [img.shape[-1], img.shape[-2]]
    raise TypeError("Unexpected type {}".format(type(img)))


def _get_inverse_affine_matrix(
        center: List[float], angle: float, translate: List[float], scale: float, shear: List[float]
) -> List[float]:
    # From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
    # Helper method to compute inverse matrix for affine transformation

    # As it is explained in PIL.Image.rotate
    # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #
    # where R is a rotation matrix, S is a scaling matrix, and SHx and SHy are the shears:
    # SHx(s) = [1, -tan(s)] and SHy(s) = [1      , 0]
    #          [0, 1      ]              [-tan(s), 1]
    #
    # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    cx, cy = center
    tx, ty = translate

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)

    # Inverted rotation matrix with scale and shear
    # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
    matrix = [d, -b, 0.0, -c, a, 0.0]
    matrix = [x / scale for x in matrix]

    # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
    matrix[2] += matrix[0] * (-cx - tx) + matrix[1] * (-cy - ty)
    matrix[5] += matrix[3] * (-cx - tx) + matrix[4] * (-cy - ty)

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += cx
    matrix[5] += cy

    return matrix


def _assert_grid_transform_inputs(
        img: Tensor,
        matrix: Optional[List[float]],
        resample: int,
        fillcolor: Optional[int],
        _interpolation_modes: Dict[int, str],
        coeffs: Optional[List[float]] = None,
):
    # From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
    if not (isinstance(img, torch.Tensor) and _is_tensor_a_torch_image(img)):
        raise TypeError("img should be Tensor Image. Got {}".format(type(img)))

    if matrix is not None and not isinstance(matrix, list):
        raise TypeError("Argument matrix should be a list. Got {}".format(type(matrix)))

    if matrix is not None and len(matrix) != 6:
        raise ValueError("Argument matrix should have 6 float values")

    if coeffs is not None and len(coeffs) != 8:
        raise ValueError("Argument coeffs should have 8 float values")

    if fillcolor is not None:
        warnings.warn("Argument fill/fillcolor is not supported for Tensor input. Fill value is zero")

    if resample not in _interpolation_modes:
        raise ValueError("Resampling mode '{}' is unsupported with Tensor input".format(resample))


def _compute_output_size(matrix: List[float], w: int, h: int) -> Tuple[int, int]:
    # From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
    # Inspired of PIL implementation:
    # https://github.com/python-pillow/Pillow/blob/11de3318867e4398057373ee9f12dcb33db7335c/src/PIL/Image.py#L2054

    # pts are Top-Left, Top-Right, Bottom-Left, Bottom-Right points.
    pts = torch.tensor([
        [-0.5 * w, -0.5 * h, 1.0],
        [-0.5 * w, 0.5 * h, 1.0],
        [0.5 * w, 0.5 * h, 1.0],
        [0.5 * w, -0.5 * h, 1.0],
    ])
    theta = torch.tensor(matrix, dtype=torch.float).reshape(1, 2, 3)
    new_pts = pts.view(1, 4, 3).bmm(theta.transpose(1, 2)).view(4, 2)
    min_vals, _ = new_pts.min(dim=0)
    max_vals, _ = new_pts.max(dim=0)

    # Truncate precision to 1e-4 to avoid ceil of Xe-15 to 1.0
    tol = 1e-4
    cmax = torch.ceil((max_vals / tol).trunc_() * tol)
    cmin = torch.floor((min_vals / tol).trunc_() * tol)
    size = cmax - cmin
    return int(size[0]), int(size[1])


def _gen_affine_grid(
        theta: Tensor, w: int, h: int, ow: int, oh: int,
) -> Tensor:
    # From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
    # https://github.com/pytorch/pytorch/blob/74b65c32be68b15dc7c9e8bb62459efbfbde33d8/aten/src/ATen/native/
    # AffineGridGenerator.cpp#L18
    # Difference with AffineGridGenerator is that:
    # 1) we normalize grid values after applying theta
    # 2) we can normalize by other image size, such that it covers "extend" option like in PIL.Image.rotate

    d = 0.5
    base_grid = torch.empty(1, oh, ow, 3, dtype=theta.dtype, device=theta.device)
    base_grid[..., 0].copy_(torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow))
    base_grid[..., 1].copy_(torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh).unsqueeze_(-1))
    base_grid[..., 2].fill_(1)

    rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)
    output_grid = base_grid.view(1, oh * ow, 3).bmm(rescaled_theta)
    return output_grid.view(1, oh, ow, 2)


def _apply_grid_transform(img: Tensor, grid: Tensor, mode: str) -> Tensor:
    # From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
    # make image NCHW
    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype != grid.dtype:
        need_cast = True
        img = img.to(grid)

    if img.shape[0] > 1:
        # Apply same grid to a batch of images
        grid = grid.expand(img.shape[0], grid.shape[1], grid.shape[2], grid.shape[3])
    img = grid_sample(img, grid, mode=mode, padding_mode="zeros", align_corners=False)

    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        # it is better to round before cast
        img = torch.round(img).to(out_dtype)

    return img


def rotate(
        img: Tensor, angle: float, resample: int = 0, expand: bool = False,
        center: Optional[List[int]] = None, fill: Optional[int] = None
) -> Tensor:
    """
    Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
    Rotate the image by angle.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    Args:
        img (PIL Image or Tensor): image to be rotated.
        angle (float or int): rotation angle value in degrees, counter-clockwise.
        resample (``PIL.Image.NEAREST`` or ``PIL.Image.BILINEAR`` or ``PIL.Image.BICUBIC``, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (list or tuple, optional): Optional center of rotation. Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.
            This option is not supported for Tensor input. Fill value for the area outside the transform in the output
            image is always 0.
    Returns:
        PIL Image or Tensor: Rotated image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """
    if not isinstance(angle, (int, float)):
        raise TypeError("Argument angle should be int or float")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    # if not isinstance(img, torch.Tensor):
    #    return F_pil.rotate(img, angle=angle, resample=resample, expand=expand, center=center, fill=fill)

    center_f = [0.0, 0.0]
    if center is not None:
        img_size = _get_image_size(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, img_size)]

    # due to current incoherence of rotation angle direction between affine and rotate implementations
    # we need to set -angle.
    matrix = _get_inverse_affine_matrix(center_f, -angle, [0.0, 0.0], 1.0, [0.0, 0.0])
    # return F_t.rotate(img, matrix=matrix, resample=resample, expand=expand, fill=fill)
    return rotate_tensor(img, matrix=matrix, resample=resample, expand=expand, fill=fill)


# def rotate(
def rotate_tensor(
        img: Tensor, matrix: List[float], resample: int = 0, expand: bool = False, fill: Optional[int] = None
) -> Tensor:
    """
    Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
    PRIVATE METHOD. Rotate the Tensor image by angle.
    .. warning::
        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.
    Args:
        img (Tensor): image to be rotated.
        matrix (list of floats): list of 6 float values representing inverse matrix for rotation transformation.
            Translation part (``matrix[2]`` and ``matrix[5]``) should be in pixel coordinates.
        resample (int, optional): An optional resampling filter. Default is nearest (=0). Other supported values:
            bilinear(=2).
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        fill (n-tuple or int or float): this option is not supported for Tensor input.
            Fill value for the area outside the transform in the output image is always 0.
    Returns:
        Tensor: Rotated image.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """
    _interpolation_modes = {
        0: "nearest",
        2: "bilinear",
    }

    _assert_grid_transform_inputs(img, matrix, resample, fill, _interpolation_modes)
    w, h = img.shape[-1], img.shape[-2]
    ow, oh = _compute_output_size(matrix, w, h) if expand else (w, h)
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    theta = torch.tensor(matrix, dtype=dtype, device=img.device).reshape(1, 2, 3)
    # grid will be generated on the same device as theta and img
    grid = _gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)
    mode = _interpolation_modes[resample]

    return _apply_grid_transform(img, grid, mode)


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """
    From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
    PRIVATE METHOD. Crop the given Image Tensor.
    .. warning::
        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.
    Args:
        img (Tensor): Image to be cropped in the form [..., H, W]. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    Returns:
        Tensor: Cropped image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError("tensor is not a torch image.")

    return img[..., top:top + height, left:left + width]


def resize(img: Tensor, size: List[int], interpolation: int = 2) -> Tensor:
    r"""
    From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
    PRIVATE METHOD. Resize the input Tensor to the given size.
    .. warning::
        Module ``transforms.functional_tensor`` is private and should not be used in user application.
        Please, consider instead using methods from `transforms.functional` module.
    Args:
        img (Tensor): Image to be resized.
        size (int or tuple or list): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
            In torchscript mode padding as a single int is not supported, use a tuple or
            list of length 1: ``[size, ]``.
        interpolation (int, optional): Desired interpolation. Default is bilinear (=2). Other supported values:
            nearest(=0) and bicubic(=3).
    Returns:
        Tensor: Resized image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError("tensor is not a torch image.")

    if not isinstance(size, (int, tuple, list)):
        raise TypeError("Got inappropriate size arg")
    if not isinstance(interpolation, int):
        raise TypeError("Got inappropriate interpolation arg")

    _interpolation_modes = {
        0: "nearest",
        2: "bilinear",
        3: "bicubic",
    }

    if interpolation not in _interpolation_modes:
        raise ValueError("This interpolation mode is unsupported with Tensor input")

    if isinstance(size, tuple):
        size = list(size)

    if isinstance(size, list) and len(size) not in [1, 2]:
        raise ValueError("Size must be an int or a 1 or 2 element tuple/list, not a "
                         "{} element tuple/list".format(len(size)))

    w, h = _get_image_size(img)

    if isinstance(size, int):
        size_w, size_h = size, size
    elif len(size) < 2:
        size_w, size_h = size[0], size[0]
    else:
        size_w, size_h = size[1], size[0]  # Convention (h, w)

    if isinstance(size, int) or len(size) < 2:
        if w < h:
            size_h = int(size_w * h / w)
        else:
            size_w = int(size_h * w / h)

        if (w <= h and w == size_w) or (h <= w and h == size_h):
            return img

    # make image NCHW
    need_squeeze = False
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    mode = _interpolation_modes[interpolation]

    out_dtype = img.dtype
    need_cast = False
    if img.dtype not in (torch.float32, torch.float64):
        need_cast = True
        img = img.to(torch.float32)

    # Define align_corners to avoid warnings
    align_corners = False if mode in ["bilinear", "bicubic"] else None

    img = torch.nn.functional.interpolate(img, size=(size_h, size_w), mode=mode, align_corners=align_corners)

    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if mode == "bicubic":
            img = img.clamp(min=0, max=255)
        img = img.to(out_dtype)

    return img


def resized_crop(
        img: Tensor, top: int, left: int, height: int, width: int, size: List[int], interpolation: int = Image.BILINEAR
) -> Tensor:
    """
    From: https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional.py
    Crop the given image and resize it to desired size.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions
    Notably used in :class:`~torchvision.transforms.RandomResizedCrop`.
    Args:
        img (PIL Image or Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
        size (sequence or int): Desired output size. Same semantics as ``resize``.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    Returns:
        PIL Image or Tensor: Cropped image.
    """
    img = crop(img, top, left, height, width)
    img = resize(img, size, interpolation)
    return img
