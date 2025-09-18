import copy
import os
import utils
import eval_utils
import sa_utils
import argparse
import joblib
from tqdm import tqdm
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from custom_types import Basic, TrainConfig
from modules import MATconv as MAT
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import matplotlib.pyplot as plt
import logging
from loss_func import BERT4NILMLoss

logging.getLogger('matplotlib').setLevel(logging.INFO)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--hidden", type=int, default=32, help="encoder decoder hidden size")
    parser.add_argument("--logname", action="store", default='root', help="name for log")
    parser.add_argument("--subName", action="store", type=str, default='test', help="name of the directory of current run")
    parser.add_argument("--inputLength", type=int, default=864, help="input length for the model")
    parser.add_argument("--outputLength", type=int, default=864, help="output length for the model")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--dataAug", action="store_true", help="data augmentation mode")
    parser.add_argument("--prob0", type=float, default=0.3, help="augment probability for Dishwasher")
    parser.add_argument("--prob1", type=float, default=0.6, help="weight")
    parser.add_argument("--prob2", type=float, default=0.3, help="weight")
    parser.add_argument("--prob3", type=float, default=0.3, help="weight")
    parser.add_argument("--test", action="store_true", help="skip training and only run evaluation")
    parser.add_argument("--show_graph", action="store_true", help="plot prediction")
    return parser.parse_args()

def train(t_net, train_Dataloader, vali_Dataloader, config, criterion, modelDir, epo=200):
    iter_loss = []
    vali_loss = []
    early_stopping_all = utils.EarlyStopping(logger, patience=20, verbose=True)

    if config.dataAug:
        sigClass = sa_utils.sigGen(config)

    path_all = os.path.join(modelDir, "All_best_onoff.ckpt")

    for e_i in range(epo):
        logger.info(f"# of epoches: {e_i}")
        for t_i, (_, _, X_scaled, Y_scaled, Y_of) in enumerate(tqdm(train_Dataloader)):
            if config.dataAug:
                X_scaled, Y_scaled, Y_of = sa_utils.dataAug(X_scaled.clone(), Y_scaled.clone(), Y_of.clone(), sigClass, config)

            t_net.model_opt.zero_grad(set_to_none=True)

            X_scaled = X_scaled.type(torch.FloatTensor).to(device, non_blocking=True)
            Y_scaled = Y_scaled.type(torch.FloatTensor).to(device, non_blocking=True)
            Y_of = Y_of.type(torch.FloatTensor).to(device, non_blocking=True)

            y_pred_dish_r, y_pred_dish_c = t_net.model(X_scaled)

            # OLD LOSS
            # loss_r = criterion[0](y_pred_dish_r,Y_scaled)
            # loss_c = criterion[1](y_pred_dish_c, Y_of)

            # loss=loss_r+loss_c
            # loss.backward()

            loss = criterion(y_pred_dish_r, Y_scaled, y_pred_dish_c, Y_of)
            loss.backward()

            t_net.model_opt.step()
            iter_loss.append(loss.item())

        epoch_losses = np.average(iter_loss)

        logger.info(f"Validation: ")
        maeScore, y_vali_ori, y_vali_pred_d_update, y_vali_ori_c, y_vali_pred_c, _ = eval_utils.evaluateResult(net, config, vali_Dataloader, logger)
        # val_loss = criterion[0](y_vali_ori, y_vali_pred_d_update)
        val_loss = criterion(y_vali_pred_d_update, y_vali_ori, y_vali_pred_c, y_vali_ori_c)
        logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses:3.3f}, val loss: {val_loss:3.3f}.")
        vali_loss.append(val_loss)

        if e_i % 10 == 0:
            checkpointName = os.path.join(modelDir, "checkpoint_" + str(e_i) + '.ckpt')
            utils.saveModel(logger, net, checkpointName)

        logger.info(f"Early stopping overall: ")
        early_stopping_all(np.mean(maeScore), net, path_all)
        if early_stopping_all.early_stop:
            print("Early stopping")
            break

    net_all = copy.deepcopy(net)
    checkpoint_all = torch.load(path_all, map_location=device)
    utils.loadModel(logger, net_all, checkpoint_all)
    net_all.model.eval()
    return net_all

if __name__ == '__main__':
    args = get_args()
    utils.mkdir("log/" + args.subName)
    logger = utils.setup_log(args.subName, args.logname)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using computation device: {device}")
    logger.info(args)
    if args.debug:
        epo = 2
    else:
        epo = 200

    # Dataloder
    logger.info(f"loading data")
    train_data, val_data, test_data = utils.data_loader(args)
    # aggregate, appliance 1, appliance 2, appliance 3, appliance 4

    logger.info(f"loading data finished")

    config_dict = {
        "input_size": 1,
        "batch_size": args.batch,
        "hidden": args.hidden,
        "lr": args.lr,
        "dropout": args.dropout,
        "logname": args.logname,
        "outputLength": args.outputLength,
        "inputLength" : args.inputLength,
        "subName": args.subName,
        "dataAug": args.dataAug,
        "prob0": args.prob0,
        "prob1": args.prob1,
        "prob2": args.prob2,
        "prob3": args.prob3,
    }

    config = TrainConfig.from_dict(config_dict)
    modelDir = utils.mkdirectory(config.subName, saveModel=True)
    joblib.dump(config, os.path.join(modelDir, "config.pkl"))

    logger.info(f"Training size: {train_data.cumulative_sizes[-1]:d}.")

    index = np.arange(0,train_data.cumulative_sizes[-1])
    train_subsampler = torch.utils.data.SubsetRandomSampler(index)
    train_Dataloader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        sampler=train_subsampler,
        num_workers=4,
        pin_memory=True)

    sampler = utils.testSampler(val_data.cumulative_sizes[-1], config.outputLength)
    sampler_test = utils.testSampler(test_data.cumulative_sizes[-1], config.outputLength)
    # sampler len = 915678

    vali_Dataloader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True)

    test_Dataloader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        sampler=sampler_test,
        num_workers=4,
        pin_memory=True)

    logger.info("Initialize model")
    model = MAT(config).to(device)
    logger.info("Model MAT")

    optim = optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=config.lr)
    net = Basic(model, optim)
    # criterion_r = nn.MSELoss()
    # criterion_c = nn.BCELoss()
    # criterion = [criterion_r, criterion_c]
    criterion = BERT4NILMLoss(tau=0.1, lambda_=1.0)

    if not args.test:
        logger.info("Training start")
        net_all = train(net, train_Dataloader, vali_Dataloader, config, criterion, modelDir, epo=epo)
        logger.info("Training end")

        logger.info("validation start")
        eval_utils.evaluateResult(net_all, config, vali_Dataloader, logger)
        logger.info("test start")
        eval_utils.evaluateResult(net_all, config, test_Dataloader, logger)
    else:
        # Load model from checkpoint
        checkpointPath = './history_model/test/s0/All_best_onoff.ckpt'

        logger.info(f"Loading model from checkpoint: {checkpointPath}")
        checkpoint = torch.load(checkpointPath, map_location=device)
        utils.loadModel(logger, net, checkpoint)
        net.model.eval()

        logger.info("Running validation on loaded model")
        eval_utils.evaluateResult(net, config, vali_Dataloader, logger)
        logger.info("Running test on loaded model")
        output = eval_utils.evaluateResult(net, config, test_Dataloader, logger)
        mae, y_ori, y_pred, y_ori_c, y_pred_c, x = output
        # y_ori, y_pred, y_ori_c, y_pred_c = ['dish washer', 'fridge', 'microwave', 'wash']

        appliances = ['dish washer', 'fridge', 'microwave', 'wash']
        tp_counts = {appliance: 0 for appliance in appliances}
        tn_counts = {appliance: 0 for appliance in appliances}
        fp_counts = {appliance: 0 for appliance in appliances}
        fn_counts = {appliance: 0 for appliance in appliances}

        # Convert the tensor outputs to numpy arrays
        y_ori_np = y_ori.cpu().numpy() if torch.is_tensor(y_ori) else y_ori
        y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
        y_ori_c_np = y_ori_c.cpu().numpy() if torch.is_tensor(y_ori_c) else y_ori_c
        y_pred_c_np = y_pred_c.cpu().numpy() if torch.is_tensor(y_pred_c) else y_pred_c

        data = []

        header = "x, " + ", ".join([f"{appliance}_ori, {appliance}_pred, {appliance}_ori_c, {appliance}_pred_c" for appliance in appliances])
        data.append(header)

        for i in range(len(x)):
            line = f"{x[i]}, " + ", ".join([f"{y_ori_np[i][j]}, {y_pred_np[i][j]}, {y_ori_c_np[i][j]}, {y_pred_c_np[i][j]}" for j in range(len(appliances))])
            data.append(line)

        # Write the data to a text file
        txt_path = 'model_output.txt'
        with open(txt_path, 'w') as f:
            for line in data:
                f.write(line + "\n")

        print(f"Model output saved to {txt_path}")

        pred_threshold = 0.5
        y_pred_bin = np.array([[1 if pred > pred_threshold else 0 for pred in appliance_preds] for appliance_preds in y_pred_c])

        for i, appliance in enumerate(appliances):
            y_true = np.array([x[i] for x in y_ori_c])
            y_pred_value = y_pred_bin[:, i]
            
            tp_counts[appliance] = np.sum((y_true == 1) & (y_pred_value == 1))
            tn_counts[appliance] = np.sum((y_true == 0) & (y_pred_value == 0))
            fp_counts[appliance] = np.sum((y_true == 0) & (y_pred_value == 1))
            fn_counts[appliance] = np.sum((y_true == 1) & (y_pred_value == 0))

        for appliance in appliances:
            print(f"{appliance.capitalize()}:")
            print(f"  True Positives: {tp_counts[appliance]}")
            print(f"  True Negatives: {tn_counts[appliance]}")
            print(f"  False Positives: {fp_counts[appliance]}")
            print(f"  False Negatives: {fn_counts[appliance]}")
            print()

        if args.show_graph:
            appliances = ['Dish Washer', 'Fridge', 'Microwave', 'Wash']
            y_ori_np = y_ori.cpu().numpy() if torch.is_tensor(y_ori) else y_ori
            y_pred_np = y_pred.cpu().numpy() if torch.is_tensor(y_pred) else y_pred
            fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
            for i in range(4):
                axes[i].plot(y_ori_np[:, i], label='Original', alpha=0.7)
                axes[i].plot(y_pred_np[:, i], label='Predicted', alpha=0.7)
                axes[i].set_title(appliances[i])
                axes[i].legend()
            plt.xlabel('Sample Index')
            plt.ylabel('Value')
            plt.suptitle('Original vs Predicted Values for Each Appliance')

            # Show plot
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()

            # Plot classification results for each appliance
            y_ori_c_np = y_ori_c.cpu().numpy() if torch.is_tensor(y_ori_c) else y_ori_c
            y_pred_c_np = y_pred_c.cpu().numpy() if torch.is_tensor(y_pred_c) else y_pred_c
            y_pred_c_binary = (y_pred_c_np > pred_threshold).astype(int)
            fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
            for i in range(4):
                axes[i].plot(y_ori_c_np[:, i], label='Original Class', alpha=0.7)
                axes[i].plot(y_pred_c_binary[:, i], label='Predicted Class', alpha=0.7)
                axes[i].set_title(appliances[i])
                axes[i].legend()
            plt.xlabel('Sample Index')
            plt.ylabel('Class')
            plt.suptitle('Original vs Predicted Classification for Each Appliance')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
