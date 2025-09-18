import torch
import numpy as np
from sklearn.metrics import f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_score(y_real, y_predict, y_real_c, y_pred_c, logger):
    maeScore = np.mean(np.abs(y_predict - y_real))
    logger.info(f"MAE: {maeScore}")

    num_period = int(len(y_real) / 1200)
    diff = 0
    for i in range(num_period):
        diff += abs(np.sum(y_real[i * 1200: (i + 1) * 1200]) - np.sum(y_predict[i * 1200: (i + 1) * 1200]))
    SAE = diff / (1200 * num_period)
    logger.info(f"SAE: {SAE}")

    f1s = f1_score(y_real_c, np.round(y_pred_c))
    logger.info(f"F1: {f1s}")

    return maeScore

def evaluate_score_multi(y_real, y_predict, y_real_c, y_pred_c, logger):
    listOfAppliance = ['dish washer', 'fridge', 'microwave', 'wash']
    mapeScore = []
    for i in range(y_predict.shape[1]):
        logger.info(f"Evaluate {listOfAppliance[i]}: ")
        mapeScore.append(evaluate_score(y_real[:,i], y_predict[:,i], y_real_c[:,i], y_pred_c[:,i], logger))
    return mapeScore
    
def evaluateResult(net, config, vali_Dataloader, logger, mode=-1):
    y_vali_pred, y_vali, y_vali_ori, y_vali_pred_c, y_vali_ori_c, truex = predict(net, config, vali_Dataloader, mode=mode)
    y_vali_pred = y_vali_pred.reshape(-1,y_vali_pred.shape[-1])
    y_vali_pred_c = y_vali_pred_c.reshape(-1,y_vali_pred.shape[-1])
    y_vali_ori = y_vali_ori.reshape(-1,y_vali_pred.shape[-1])
    y_vali_ori_c = y_vali_ori_c.reshape(-1,y_vali_pred.shape[-1])
    y_vali_pred[y_vali_pred < 0] = 0
    y_vali_pred_d = y_vali_pred * 612

    if mode >= 0:
        maeScore = evaluate_score(y_vali_ori.numpy(), y_vali_pred_d.numpy(), y_vali_ori_c.numpy(), y_vali_pred_c.numpy(), logger)
    else:
        maeScore = evaluate_score_multi(y_vali_ori.numpy(), y_vali_pred_d.numpy(), y_vali_ori_c.numpy(), y_vali_pred_c.numpy(), logger)
    return maeScore, y_vali_ori, y_vali_pred_d, y_vali_ori_c, y_vali_pred_c, truex.reshape(-1, 1)

def predict(t_net, t_cfg, vali_Dataloader, mode=-1):
    y_pred_r = []
    y_true_scaled_r = []
    y_true_r = []
    y_pred_c = []
    y_true_c = []
    x_true = []
    
    start = int((t_cfg.inputLength-t_cfg.outputLength)/2)
    end = start + t_cfg.outputLength

    with torch.no_grad():
        for _, (X, Y, X_scaled, Y_scaled, Y_of) in enumerate(vali_Dataloader):
            if mode>=0:
                Y = Y[:,start:end,[mode]]
                Y_scaled = Y_scaled[:,start:end,[mode]]
                Y_of = Y_of[:,start:end,[mode]]
            else:
                Y = Y[:,start:end,:]
                Y_scaled = Y_scaled[:,start:end,:]
                Y_of = Y_of[:,start:end,:]

            X_scaled = X_scaled.type(torch.FloatTensor).to(device, non_blocking=True)
            Y_scaled = Y_scaled.type(torch.FloatTensor).to(device, non_blocking=True)
            Y = Y.type(torch.FloatTensor).to(device, non_blocking=True)
            Y_of = Y_of.type(torch.FloatTensor).to(device, non_blocking=True)

            output_r, output_c = t_net.model(X_scaled)
            y_pred_r.append(output_r.cpu())
            y_true_scaled_r.append(Y_scaled.cpu())
            y_true_r.append(Y.cpu())
            x_true.append(X[:,start:end,:])

            y_pred_c.append(output_c.cpu())
            y_true_c.append(Y_of.cpu())

        out_pred_scaled_r = torch.vstack(y_pred_r)
        out_true_scaled_r = torch.vstack(y_true_scaled_r)
        out_true_r = torch.vstack(y_true_r)
        out_true_x = torch.vstack(x_true)
        out_pred_scaled_c = torch.vstack(y_pred_c)
        out_true_c = torch.vstack(y_true_c)
    return out_pred_scaled_r, out_true_scaled_r, out_true_r, out_pred_scaled_c, out_true_c, out_true_x