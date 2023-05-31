# imports
import os
import numpy as np
import random
import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from train_process import train_epochs
from eval_process import eval_calculations

################################################################################################
################################################################################################

def seed_everything(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

################################################################################################
################################################################################################

def train_resnet50(path_to_data_train, path_to_data_test, path_output, hyperparam_configuration_resnet50):
    path_output = os.path.join(path_output, 'resnet50')
    if not os.path.exists(path_output):
        os.mkdir(path_output)
        
    # initilizationnn
    seed = 0
    seed_everything(seed)
    g = get_generator(seed)

    with open(os.path.join(path_output, "hyperparameters.txt"), 'w') as hyperparam_file:
        for key, value in hyperparam_configuration_resnet50.items():
            hyperparam_file.write('%s:%s\n' % (key, value))

    transform_ResNet50_train = transforms.Compose([
        transforms.Resize(hyperparam_configuration_resnet50["IMAGE_RESIZE_SIZE"], interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_ResNet50_test = transforms.Compose([
        transforms.Resize(hyperparam_configuration_resnet50["IMAGE_RESIZE_SIZE"], interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # data
    ds_train = datasets.ImageFolder(path_to_data_train, transform=transform_ResNet50_train)
    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=hyperparam_configuration_resnet50["BATCH_SIZE"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    ds_test = datasets.ImageFolder(path_to_data_test, transform=transform_ResNet50_test)
    testloader = torch.utils.data.DataLoader(ds_test, batch_size=hyperparam_configuration_resnet50["BATCH_SIZE"], worker_init_fn=seed_worker, generator=g)
    dataloaders = {'train':trainloader, 'test':testloader}
    dataset_sizes = {'train':len(ds_train), 'test':len(ds_test)}
    
    # model definition
    # pretrained ResNet50
    model_pretrained = torchvision.models.resnet50(pretrained=True)
    model = model_pretrained
    
    for param in model.parameters():
        param.requires_grad = hyperparam_configuration_resnet50["PARAM_REQUIRES_GRAD"]

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, hyperparam_configuration_resnet50["NUMBER_ENTITIES"])
    
    print(model)
    model = nn.DataParallel(model, device_ids=[0,1,2])
    model = model.to(f'cuda:{model.device_ids[0]}')

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=hyperparam_configuration_resnet50["LEARNING_RATE"], momentum=hyperparam_configuration_resnet50["OPTIMIZER_MOMENTUM"])

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    num_epochs = hyperparam_configuration_resnet50["NUMBER_EPOCHS"]
    model = train_epochs(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, dataloaders, dataset_sizes, path_output)
    file_name = 'resnet50_ep_' + str(num_epochs) + '.pth'
    torch.save(model, os.path.join(path_output, file_name))
    file_name = 'resnet50_ep_' + str(num_epochs) + '_state.pth'
    torch.save(model.state_dict(), os.path.join(path_output, file_name))

    # evaluation
    class_names = ds_train.classes
    eval_calculations(model, testloader, class_names, path_output)

    return model

################################################################################################
################################################################################################

def train_visiontransformer(path_to_data_train, path_to_data_test, path_output, hyperparam_configuration_visiontrans):
    path_output = os.path.join(path_output, 'visiontransformer')
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    
    # initilization
    seed = 0
    seed_everything(seed)
    g = get_generator(seed)
    
    with open(os.path.join(path_output, "hyperparameters.txt"), 'w') as hyperparam_file:
        for key, value in hyperparam_configuration_visiontrans.items():
            hyperparam_file.write('%s:%s\n' % (key, value))

    transform_visiontrans_train = transforms.Compose([
        transforms.Resize(hyperparam_configuration_visiontrans["IMAGE_RESIZE_SIZE"], interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_visiontrans_test = transforms.Compose([
        transforms.Resize(hyperparam_configuration_visiontrans["IMAGE_RESIZE_SIZE"], interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # data
    ds_train = datasets.ImageFolder(path_to_data_train, transform=transform_visiontrans_train)
    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=hyperparam_configuration_visiontrans["BATCH_SIZE"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    ds_test = datasets.ImageFolder(path_to_data_test, transform=transform_visiontrans_test)
    testloader = torch.utils.data.DataLoader(ds_test, batch_size=hyperparam_configuration_visiontrans["BATCH_SIZE"], worker_init_fn=seed_worker, generator=g)
    dataloaders = {'train':trainloader, 'test':testloader}
    dataset_sizes = {'train':len(ds_train), 'test':len(ds_test)}

    # model definition
    model_pretrained = torchvision.models.vit_b_16(pretrained=True)
    model = model_pretrained
    
    for param in model.parameters():
        param.requires_grad = hyperparam_configuration_visiontrans["PARAM_REQUIRES_GRAD"]

    num_ftrs = model.heads.head.in_features
    model.heads.head = nn.Linear(num_ftrs, hyperparam_configuration_visiontrans["NUMBER_ENTITIES"])
    print(model)
    model = nn.DataParallel(model, device_ids=[0,1,2])
    model = model.to(f'cuda:{model.device_ids[0]}') 

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=hyperparam_configuration_visiontrans["LEARNING_RATE"], momentum=hyperparam_configuration_visiontrans["OPTIMIZER_MOMENTUM"])

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    num_epochs = hyperparam_configuration_visiontrans["NUMBER_EPOCHS"]
    model = train_epochs(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, dataloaders, dataset_sizes, path_output)
    file_name = 'visiontranformer_ep_' + str(num_epochs) + '.pth'
    torch.save(model, os.path.join(path_output, file_name))
    file_name = 'visiontranformer_ep_' + str(num_epochs) + '_state.pth'
    torch.save(model.state_dict(), os.path.join(path_output, file_name))
    
    # evaluation
    class_names = ds_train.classes
    eval_calculations(model, testloader, class_names, path_output)

    return model

################################################################################################
################################################################################################

def train_convnext(path_to_data_train, path_to_data_test, path_output, hyperparam_configuration_convnext):
    path_output = os.path.join(path_output, 'convnext')
    if not os.path.exists(path_output):
        os.mkdir(path_output)
        
    # initilization
    seed = 0
    seed_everything(seed)
    g = get_generator(seed)

    with open(os.path.join(path_output, "hyperparameters.txt"), 'w') as hyperparam_file:
        for key, value in hyperparam_configuration_convnext.items():
            hyperparam_file.write('%s:%s\n' % (key, value))

    transform_convnext_train = transforms.Compose([
        transforms.Resize(hyperparam_configuration_convnext["IMAGE_RESIZE_SIZE"], interpolation=InterpolationMode.BILINEAR),
        transforms.ColorJitter(saturation=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_convnext_test = transforms.Compose([
        transforms.Resize(hyperparam_configuration_convnext["IMAGE_RESIZE_SIZE"], interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # data
    ds_train = datasets.ImageFolder(path_to_data_train, transform=transform_convnext_train)
    trainloader = torch.utils.data.DataLoader(ds_train, batch_size=hyperparam_configuration_convnext["BATCH_SIZE"], shuffle=True, worker_init_fn=seed_worker, generator=g)
    ds_test = datasets.ImageFolder(path_to_data_test, transform=transform_convnext_test)
    testloader = torch.utils.data.DataLoader(ds_test, batch_size=hyperparam_configuration_convnext["BATCH_SIZE"], worker_init_fn=seed_worker, generator=g)
    dataloaders = {'train':trainloader, 'test':testloader}
    dataset_sizes = {'train':len(ds_train), 'test':len(ds_test)} 
    
    # model definition
    model_pretrained = torchvision.models.convnext_base(pretrained=True)
    model = model_pretrained
    
    for param in model.parameters():
        param.requires_grad = hyperparam_configuration_convnext["PARAM_REQUIRES_GRAD"]
    
    model.classifier[2] = nn.Linear(in_features=model.classifier[2].in_features, out_features=hyperparam_configuration_convnext["NUMBER_ENTITIES"])
    print(model)
    model = nn.DataParallel(model, device_ids=[0,1,2])
    model = model.to(f'cuda:{model.device_ids[0]}')

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model.parameters(), lr=hyperparam_configuration_convnext["LEARNING_RATE"], momentum=hyperparam_configuration_convnext["OPTIMIZER_MOMENTUM"])

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    num_epochs = hyperparam_configuration_convnext["NUMBER_EPOCHS"]
    model = train_epochs(model, criterion, optimizer_ft, exp_lr_scheduler, num_epochs, dataloaders, dataset_sizes, path_output)
    file_name = 'convnext_ep_' + str(num_epochs) + '.pth'
    torch.save(model, os.path.join(path_output, file_name))
    file_name = 'convnext_ep_' + str(num_epochs) + '_state.pth'
    torch.save(model.state_dict(), os.path.join(path_output, file_name))

    # evaluation
    class_names = ds_train.classes
    eval_calculations(model, testloader, class_names, path_output)
    
    return model