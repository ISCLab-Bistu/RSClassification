import numpy as np
import torch
from rmsm import Config
from torch.autograd import Variable
import torch as t

from rmsm.apis.inference import init_model, inference_model
from rmsm.datasets import build_dataset, build_dataloader
from rmsm.models import build_backbone


# Get predicted value
def get_predictions(model, dataloader, cuda, get_probs=False):
    preds = []
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        targets = targets.long()
        outputs = model(inputs)
        print(outputs)
        if get_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1)
            print(probs)
            if cuda:
                probs = probs.data.cpu().numpy()
            else:
                probs = probs.data.numpy()
            preds.append(probs)
        else:
            probability, predicted = torch.max(outputs.data, 1)
            if cuda: predicted = predicted.cpu()
            preds += list(predicted.numpy().ravel())
    if get_probs:
        return np.vstack(preds)
    else:
        return np.array(preds)


def main(path):
    test = []
    for i in range(99, 100):
        # checkpoint_path = 'formic_acid/epoch_' + str(i + 1) + '.pth'
        checkpoint_path = 'Microplastics/checkpoint/chechpoint.pt'
        # Initialize the configuration file according to the weights
        model = init_model(path, checkpoint_path, device='cpu')  # or device='cuda:0'
        cfg = Config.fromfile(path)  # The corresponding configuration file is obtained after parsing

        # build dataset
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset, samples_per_gpu=32, workers_per_gpu=2, dist=False, shuffle=False)

        result = get_predictions(model, data_loader, cuda=False, get_probs=True)
        print(result)


if __name__ == '__main__':
    path = "../../configs/resnet/raman_Microplastics.py"
    main(path)
