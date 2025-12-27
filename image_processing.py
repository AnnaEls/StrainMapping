def extract_patches(image, patch_size:int=128, stride:int=64):
  """
  Extract non-overlapping patches from an image.
  Parameters:
  - image: input image
  - patch_size: size of each patch
  - stride: stride between patches
  Returns:
  - patches: list of extracted patches
  - coords: list of coordinates of the top-left corner of each patch (y, x)
  """
  patches = []
  coords = []

  H, W = image.shape
  for y0 in range(0, H - patch_size + 1, stride):
      for x0 in range(0, W - patch_size + 1, stride):
          patch = image[y0:y0+patch_size, x0:x0+patch_size]
          patches.append(patch)
          coords.append((y0, x0))
  return patches, coords