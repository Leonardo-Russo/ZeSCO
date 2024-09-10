# from dinov2.dinov2.eval.depth.models import build_depther


# class CenterPadding(torch.nn.Module):
#     def __init__(self, multiple):
#         super().__init__()
#         self.multiple = multiple

#     def _get_pad(self, size):
#         new_size = math.ceil(size / self.multiple) * self.multiple
#         pad_size = new_size - size
#         pad_size_left = pad_size // 2
#         pad_size_right = pad_size - pad_size_left
#         return pad_size_left, pad_size_right

#     @torch.inference_mode()
#     def forward(self, x):
#         pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
#         output = F.pad(x, pads)
#         return output


# def create_depther(cfg, backbone_model, backbone_size, head_type):
#     train_cfg = cfg.get("train_cfg")
#     test_cfg = cfg.get("test_cfg")
#     depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

#     depther.backbone.forward = partial(
#         backbone_model.get_intermediate_layers,
#         n=cfg.model.backbone.out_indices,
#         reshape=True,
#         return_class_token=cfg.model.backbone.output_cls_token,
#         norm=cfg.model.backbone.final_norm,
#     )

#     if hasattr(backbone_model, "patch_size"):
#         depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

#     return depther

# def make_depth_transform() -> transforms.Compose:
#     return transforms.Compose([
#         transforms.ToTensor(),
#         lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
#         transforms.Normalize(
#             mean=(123.675, 116.28, 103.53),
#             std=(58.395, 57.12, 57.375),
#         ),
#     ])


# def render_depth(values, colormap_name="magma_r") -> Image:
#     min_value, max_value = values.min(), values.max()
#     normalized_values = (values - min_value) / (max_value - min_value)

#     colormap = matplotlib.colormaps[colormap_name]
#     colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
#     colors = colors[:, :, :3] # Discard alpha component
#     return Image.fromarray(colors)



# def apply_depth_estimation(image, device, debug=False):
#     """
#     Applies DINOv2 depth estimator to the image.

#     Args:
#     - image: The input image.
#     - device: The device to run the model on ('cuda' or 'cpu').
#     - debug: Whether to print debugging information and display the depth map.

#     Returns:
#     - depth_map: The estimated depth map of the image.
#     - foreground, middleground, background: Boolean masks for pixel classification.
#     """

#     # Load pretrained depth head (adapt this part from the notebook)
#     # ... (Include the code from the notebook to load the depth head and create the depth estimator)

#     # Preprocess the image using the DINOv2 transform
#     transform = make_depth_transform()  # Assuming you have this function from the notebook
#     scale_factor = 1  # Adjust if needed
#     rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
#     transformed_image = transform(rescaled_image)
#     batch = transformed_image.unsqueeze(0).to(device)

#     # Estimate depth
#     with torch.inference_mode():
#         result = model.whole_inference(batch, img_meta=None, rescale=True)
#         depth_map = result.squeeze().cpu().numpy()

#     # Normalize depth map to [0, 1]
#     depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

#     # Classify pixels as foreground, middleground, or background
#     depth_thresholds = np.percentile(depth_map, [33, 66])
#     foreground = depth_map <= depth_thresholds[0]
#     middleground = (depth_map > depth_thresholds[0]) & (depth_map <= depth_thresholds[1])
#     background = depth_map > depth_thresholds[1]

#     if debug:
#         print("depth map shape: ", depth_map.shape)
#         plt.figure(figsize=(8, 8))
#         plt.imshow(depth_map, cmap='plasma')
#         plt.title("Depth Map")
#         plt.axis('off')
#         plt.show()

#     return depth_map, foreground, middleground, background