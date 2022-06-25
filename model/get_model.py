import timm


def get_model(args, num_classes):
    print('==> Building model..')
    if args.net == 'resnet18':
        net = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)
    elif args.net == 'resnet34':
        net = timm.create_model('resnet34', pretrained=True, num_classes=num_classes)
    elif args.net == 'resnet50':
        net = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
    elif args.net == 'wrn50-2':
        net = timm.create_model('wide_resnet50_2', pretrained=True, num_classes=num_classes)
    elif args.net == 'wrn101-2':
        net = timm.create_model('wide_resnet101_2', pretrained=True, num_classes=num_classes)
    elif args.net == 'inc_v3':
        net = timm.create_model('inception_v3', pretrained=True, num_classes=num_classes)
    elif args.net == 'inc_v4':
        net = timm.create_model('inception_v4', pretrained=True, num_classes=num_classes)
    elif args.net == 'bit50-3':
        net = timm.create_model('resnetv2_50x3_bitm', pretrained=True, num_classes=num_classes)
    elif args.net == 'bit101-3':
        net = timm.create_model('resnetv2_101x3_bitm', pretrained=True, num_classes=num_classes)
    elif args.net == 'bit152-4':
        net = timm.create_model('resnetv2_152x4_bitm', pretrained=True, num_classes=num_classes)
    elif args.net == 'vit-b':
        net = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    elif args.net == 'vit-s':
        net = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=num_classes)
    elif args.net == 'vit-t':
        net = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
    elif args.net == 'deit-b':
        net = timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=num_classes)
    elif args.net == 'deit-s':
        net = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=num_classes)
    elif args.net == 'deit-t':
        net = timm.create_model('deit_tiny_patch16_224', pretrained=True, num_classes=num_classes)
    elif args.net == 'swin-b':
        net = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif args.net == 'swin-s':
        net = timm.create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=num_classes)
    elif args.net == 'swin-t':
        net = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=num_classes)
    else:
        raise 'no matched model'

    return net