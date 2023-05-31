import copy
import os
import time
import torch

################################################################################################

def train_epochs(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, path_output):    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        filename = os.path.join(path_output, 'log.txt')
        f = open(filename, 'a')
        print(f'Epoch {epoch}/{num_epochs - 1}', file=f)
        print('-' * 10, file=f)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs_gpu = inputs.to(f'cuda:{model.device_ids[0]}')
                labels_gpu = labels.to(f'cuda:{model.device_ids[0]}')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs_gpu)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels_gpu)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()                    

                # statistics
                running_loss += loss.item() * inputs_gpu.size(0)
                running_corrects += torch.sum(preds == labels_gpu.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}', file=f)

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
        print(' ', file=f)
        f.close()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    filename = os.path.join(path_output, 'log.txt')
    f = open(filename, 'a')
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s', file=f)
    print(f'Best val Acc: {best_acc:4f}', file=f)
    f.close()

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
