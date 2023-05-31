import os
import torch

from sklearn import metrics

################################################################################################

def eval_calculations(model, testloader, class_names, path_output):   
    y_pred = []
    y_true = []
    y_score = []

    # iterate over test data
    for inputs, labels in testloader:
        inputs = inputs.to(f'cuda:{model.device_ids[0]}')
        labels = labels.to(f'cuda:{model.device_ids[0]}')
        
        output = model(inputs) # Feed Network
        y_score.extend(output.data.cpu().numpy())

        output_max = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output_max) # Save Prediction
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

    # create metrics value    
    filename = os.path.join(path_output, 'results.txt')
    f = open(filename, 'a')
    print('accuracy_score', file=f)
    res = metrics.accuracy_score(y_true, y_pred)
    print(res, file=f)
    print('f1_score - macro', file=f)
    res = metrics.f1_score(y_true, y_pred, average='macro')
    print(res, file=f)
    print('precision_score - macro', file=f)
    res = metrics.precision_score(y_true, y_pred, average='macro')
    print(res, file=f)
    print('recall_score - macro', file=f)
    res = metrics.recall_score(y_true, y_pred, average='macro')
    print(res, file=f)
    f.close()