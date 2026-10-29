# =============================================================================
# 尾矿库(weikuang) 单类别 OBB 数据集 —— MMRotate DOTADataset 配置
# 放置路径: mmrotate/configs/_base_/datasets/dotav1_tailings.py
# 由 make_dota_split.py 生成的目录结构:
#   data/split_ss_dota/{train,val,test}/{images,annfiles}/
# =============================================================================

# 单类别: 尾矿库
dataset_type = 'DOTADataset'
data_root = 'data/tailing/'
classes = ('tailing_pond', )

backend_args = None

# ---------------------------------------------------------------------
# 注意: LoadAnnotations / ConvertBoxType 必须放在 Resize 之前!
# 否则对于少数 >1024 的大切片, 图像被缩放而 bbox 没被缩放, 框会整体错位。
# (原 dotav1 配置里那句 "avoid bboxes being resized" 只在原图恰好等于
#  1024 时才不出问题, 这里有 36 张更大的切片, 必须改顺序)
# ---------------------------------------------------------------------
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.75,
         direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='RandomRotate', prob=0.5, angle_range=180),
    # 新增 2: 平移变换 (旋转框友好: 边框用 pad_val 填充, 中心移出图像的框会被剔除)
    dict(type='RandomTranslate', prob=0.5, max_shift_px=100, pad_val=114),
    # 新增 3: 高斯噪声引入
    # 注意: 必须放在 PhotoMetricDistortion 之前 —— 此时图像仍是 uint8(0-255),
    # sigma 用 0-255 尺度更直观; PhotoMetricDistortion 之后图像会变成 float32。
    # dict(type='RandomGaussianNoise', prob=0.5, mean=0.0, sigma=(5.0, 15.0)),
    #新增 1: 亮度/对比度/饱和度/色相
    dict(type='mmdet.PhotoMetricDistortion',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='mmdet.Pad', size=(1024, 1024),
         pad_val=dict(img=(114, 114, 114))),
    dict(type='mmdet.PackDetInputs')
]

val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = val_pipeline

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    pin_memory=False,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        test_mode=True,
        metainfo=dict(classes=classes),
        pipeline=val_pipeline))

val_evaluator = dict(type='DOTAMetric', metric='mAP')

# ---------------------------------------------------------------------
# test: 本数据集是有标注的(70/10/20 划分出来的), 所以 test 也要算指标,
# 不要开 format_only / merge_patches(那是给 DOTA 官方无标注测试集提交用的)
# ---------------------------------------------------------------------
test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/annfiles/',
        data_prefix=dict(img_path='test/images/'),
        test_mode=True,
        metainfo=dict(classes=classes),
        pipeline=test_pipeline))

test_evaluator = dict(type='DOTAMetric', metric='mAP')
