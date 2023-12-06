import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from dataset_cifar10 import Cifar10
from model_alexnet import AlexNet
from utils import get_uncertain_samples, get_high_confidence_samples,update_threshold
import argparse

#Learning algorithm of CEAL
def ceal_learning_algorithm(du: DataLoader, dl: DataLoader,
                            dtest: DataLoader,
                            k: int = 1000,
                            delta_0: float = 0.000001,
                            dr: float = 0.00033,
                            t: int = 1,
                            epochs: int = 5,
                            criteria: str = 'cl',
                            max_iter: int = 2,
                            save_model:bool = False):
    """
    Parameters
    du: DataLoader, Unlabeled samples
    dl : DataLoader, labeled samples
    dtest : DataLoader, test data
    k: int, (default = 1000), uncertain samples selection
    delta_0: float, hight confidence samples selection threshold
    dr: float, threshold decay
    t: int, fine-tuning interval
    epochs: int
    criteria: str
    max_iter: int,maximum iteration number.
    """
    print('Initial configuration: len(du): {}, len(dl): {} '.format(len(du.sampler.indices),len(dl.sampler.indices)))

    for iteration in range(max_iter):

        # Create the model
        model = AlexNet(n_classes=10, device=None)

        # Initialize the model
        print('Intialize training the model on `dl` and test on `dtest`')

        model.train(epochs=epochs, train_loader=dl, valid_loader=None)

        # Evaluate model on dtest
        acc = model.evaluate(test_loader=dtest)

        print('====> Initial accuracy: {} '.format(acc))

        print('Iteration: {}: run prediction on unlabeled data ''`du` '.format(iteration))

        pred_prob = model.predict(test_loader=du)

        # get k uncertain samples
        uncert_samp_idx, _ = get_uncertain_samples(pred_prob=pred_prob, k=k,
                                                   criteria=criteria)

        # get original indices
        uncert_samp_idx = [du.sampler.indices[idx] for idx in uncert_samp_idx]

        # add the uncertain samples selected from `du` to the labeled samples
        #  set `dl`
        dl.sampler.indices.extend(uncert_samp_idx)

        print(
            'Update size of `dl`  and `du` by adding uncertain {} samples'
            ' in `dl`'
            ' len(dl): {}, len(du) {}'.
            format(len(uncert_samp_idx), len(dl.sampler.indices),
                   len(du.sampler.indices)))

        # get high confidence samples `dh`
        hcs_idx, hcs_labels = get_high_confidence_samples(pred_prob=pred_prob,
                                                          delta=delta_0)
        # get the original indices
        hcs_idx = [du.sampler.indices[idx] for idx in hcs_idx]

        # remove the samples that already selected as uncertain samples.
        hcs_idx = [x for x in hcs_idx if
                   x not in list(set(uncert_samp_idx) & set(hcs_idx))]

        # add high confidence samples to the labeled set 'dl'

        # (1) update the indices
        dl.sampler.indices.extend(hcs_idx)
        # (2) update the original labels with the pseudo labels.
        for idx in range(len(hcs_idx)):
            dl.dataset.labels[hcs_idx[idx]] = hcs_labels[idx]
        print(
            'Update size of `dl`  and `du` by adding {} hcs samples in `dl`'
            ' len(dl): {}, len(du) {}'.
            format(len(hcs_idx), len(dl.sampler.indices),
                   len(du.sampler.indices)))

        if iteration % t == 0:
            print('Iteration: {} fine-tune the model on dh U dl'.
                        format(iteration))
            model.train(epochs=epochs, train_loader=dl)

            # update delta_0
            delta_0 = update_threshold(delta=delta_0, dr=dr, t=iteration)

        # remove the uncertain samples from the original `du`
        print('remove {} uncertain samples from du'.
                    format(len(uncert_samp_idx)))
        for val in uncert_samp_idx:
            du.sampler.indices.remove(val)

        acc = model.evaluate(test_loader=dtest)
        print(
            "Iteration: {}, len(dl): {}, len(du): {},"
            " len(dh) {}, acc: {} ".format(
                iteration, len(dl.sampler.indices),
                len(du.sampler.indices), len(hcs_idx), acc))

    ##save final model checkpoint 
    if save_model:           
      model.save_checkpoint('CEAL_checkpoint.pt')

if __name__ == "__main__":
    ##adding arguements
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1000, help = "number of uncertain samples") 
    parser.add_argument("--criteria", type=str, default= 'cl', help = "cl(least confidence) or ms(margin sampling) or en(entropy)") 
    parser.add_argument("--threshold", type=float, default = 0.000001, help = 'hight confidence samples selection threshold')
    parser.add_argument("--threshold_decay", type=float, default = 0.00033)
    parser.add_argument("--epochs", type=int,default=10) 
    parser.add_argument("--max_itr", type=int,default=10) 
    parser.add_argument("--batch_size", type=int,default=16) 
    parser.add_argument("--random_seed", type=int, default=123) 
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--save_model", type=bool, default= False, help="True flag for saving model checkpoint ") 
    args = parser.parse_args()

    ##dataset
    dataset_train = Cifar10(
        root_dir="data/cifar10/train/",
        transform=transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((128,128)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]))

    dataset_test = Cifar10(
        root_dir="data/cifar10/test/",
        transform=transforms.Compose([transforms.ToPILImage(),
        transforms.Resize((128,128)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]))


    #Reading arguments. 
    k = args.k
    criteria = args.criteria
    threshold = args.threshold
    threshold_decay = args.threshold_decay
    epochs = args.epochs
    max_itr = args.max_itr
    random_seed = args.random_seed
    batch_size = args.batch_size
    validation_split = args.validation_split


    # Creating data indices for training and validation splits:
    shuffling_dataset = True
    dataset_size = len(dataset_train)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffling_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    du = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                     sampler=train_sampler, num_workers=4)
    dl = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                     sampler=valid_sampler, num_workers=4)
    dtest = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                        num_workers=4)

    ceal_learning_algorithm(du=du, dl=dl, dtest=dtest,k=k, delta_0=threshold, dr=threshold_decay, t=1,epochs=epochs, criteria=criteria, max_iter=max_itr, save_model=args.save_model)
