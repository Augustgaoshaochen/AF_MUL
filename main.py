import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random

import torch
from options import Options
# from dataset.UCR_dataloader import load_data
import numpy as np
# from MlultModal.tool.CPSC.load_data_cpsc2021 import data_2_dataset
# from tool.CPSC.load_data_cpsc2021 import load_data, data_2_dataset, load_data_ant
from tool.CPSC_dataloader.load_data_cpsc2021 import load_data, data_2_dataset, load_icentia11k_data, load_SPH_data, \
    load_vfdb_data, load_data_ptbxl
from tool.yxdecg import DECG_Noise_Dataset

device = torch.device("cuda:0" if
                      torch.cuda.is_available() else "cpu")
# torch.cuda.set_per_process_memory_fraction(0.05, 1)

opt = Options().parse()
# os.environ['CUDA_LAUNCH_BLOCKING'] = 1

if opt.model == "BeatGAN":
    from model.BeatGAN import BeatGAN as ModelTrainer
elif opt.model == "AE_CNN":
    from model.AE_CNN import ModelTrainer
elif opt.model == 'AE_LSTM':
    from model.AE_LSTM import ModelTrainer
elif opt.model == 'Ganomaly':
    from model.Ganomaly import Ganomaly as ModelTrainer
elif opt.model == 'AE_CNN_noisy_multi':
    from model.AE_CNN_noisy_multi import ModelTrainer
elif opt.model == 'AE_CNN_noise':
    from model.AE_CNN_noise import ModelTrainer
elif opt.model == 'AE_CNN_RS':
    from model.AE_CNN_RS import ModelTrainer
elif opt.model == 'AE_CNN_MASK':
    from model.AE_CNN_mask import ModelTrainer
elif opt.model == 'AE_CNN_RS_mask':
    from model.AE_CNN_RS_mask import ModelTrainer
elif opt.model == 'AE_CNN_AHLF':
    from model.AE_CNN_AHLF import ModelTrainer
elif opt.model == 'AE_CNN_Heart':
    from model.AE_CNN_Heart import ModelTrainer
elif opt.model == 'AE_CNN_Noisy_power':
    from model.AE_CNN_Noisy_power import ModelTrainer
elif opt.model == 'Deep_Mul':
    from model.Deep_Mul import ModelTrainer
elif opt.model == 'AE_CNN_AHLF_WO_AFI':
    from model.AE_CNN_AHLF_WO_AFI import ModelTrainer
else:
    raise Exception("no this model_eeg :{}".format(opt.model))

SEEDS = [1, 2, 3, 4, 5, 6]
# SEEDS = [6]

if __name__ == '__main__':

    lens = 1000
    # results_dir = './11ktestsseed123' + str(lens)
    # results_dir = './xiaorong/wo_sdnnAfi_11k' #+ str(lens)
    # results_dir = './Mask/zero/11k' #+ str(lens)
    results_dir = './vfdb'  # + str(lens)

    opt.outf = results_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file2print = open('{}/results_{}.log'.format(results_dir, opt.model), 'a+')
    file2print_detail = open('{}/results_{}_detail.log'.format(results_dir, opt.model), 'a+')

    import datetime

    print(datetime.datetime.now(), file=file2print)
    print(datetime.datetime.now(), file=file2print_detail)

    print("Model\tDataset\tNormal_Label\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print_detail)
    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch")
    print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print)
    file2print.flush()
    file2print_detail.flush()

    root = "/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL/"
    read_mk = "/home/chenpeng/workspace/dataset/CSPC2021_fanc/npy_data_10s/"
    opt.noisy_classify = 5
    opt.nc = 2
    opt.batchsize = 128

    AUCs = {}
    APs = {}
    MAX_EPOCHs = {}
    error = 1
    if lens == 400:
        opt.list_upsample = [1, 4]
    elif lens == 600:
        opt.list_upsample = [2, 3]
    elif lens == 800:
        opt.list_upsample = [2, 4]
    elif lens == 1000:
        opt.list_upsample = [2, 5]

    opt.Snr = [10, 10, 10, 10, 10]  # Gus,uniform,exponential,ray,gamma
    MAX_EPOCHs_seed = {}
    AUCs_seed = {}
    APs_seed = {}
    pres_seed = []
    recall_seed = []
    f1_seed = []
    model_result = {}
    for seed in SEEDS:
        opt.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True

        if opt.dataset == 'CPSC2021':
            train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data(
                '/home/chenpeng/workspace/dataset/CSPC2021_fanc/ALL_100HZ/', lens, seed)

        elif opt.dataset == 'icentia11k':
            opt.nc = 1
            train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_icentia11k_data(
                '/data/icentia11k/', seed)
            train_data = np.expand_dims(train_data, axis=1)
            val_normal_data = np.expand_dims(val_normal_data, axis=1)
            val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
            test_normal_data = np.expand_dims(test_normal_data, axis=1)
            test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)

        elif opt.dataset == 'SPH_data':
            opt.nc = 12
            train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_SPH_data(
                seed)
        elif opt.dataset == 'PTBXL':
            opt.nc = 12
            train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_data_ptbxl(seed)

        elif opt.dataset == 'vfdb':
            opt.nc = 1
            train_data, val_normal_data, val_abnormal_data, test_normal_data, test_abnormal_data = load_vfdb_data(seed)
            train_data = np.expand_dims(train_data, axis=1)
            val_normal_data = np.expand_dims(val_normal_data, axis=1)
            val_abnormal_data = np.expand_dims(val_abnormal_data, axis=1)
            test_normal_data = np.expand_dims(test_normal_data, axis=1)
            test_abnormal_data = np.expand_dims(test_abnormal_data, axis=1)

        opt.seed = seed
        np.random.seed(seed)
        try:
            opt.snr = 5
            dataloader, opt.isize = data_2_dataset(train_data, val_normal_data, val_abnormal_data, test_normal_data,
                                                   test_abnormal_data, opt)
            opt.isize = 1000
        except Exception as e:
            print(e)
            break
        model = ModelTrainer(opt, dataloader, device)
        # print(model.G)
        opt.name = "%s/%s" % (opt.model, opt.dataset)
        expr_dir = os.path.join(opt.outf, opt.name, 'train')
        test_dir = os.path.join(opt.outf, opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        args = vars(opt)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        print(opt)
        print("################  Train  ##################")
        ap_test, auc_test, epoch_max_point, Pre_test, Recall_test, f1_test = model.train()
        print("SEED: {}\t{}\t{}\t{}\t{}\t{}\t{}".format(seed, auc_test, ap_test, epoch_max_point, Pre_test,
                                                        Recall_test, f1_test), file=file2print)
        file2print.flush()
        AUCs_seed[seed] = auc_test
        APs_seed[seed] = ap_test
        MAX_EPOCHs_seed[seed] = epoch_max_point
        pres_seed.append(Pre_test)
        recall_seed.append(Recall_test)
        f1_seed.append(f1_seed)
        if opt.model == "Ganomaly":
            seed_index = "SEED" + str(seed)
            model_result[seed_index] = {}
            model_result[seed_index]['Generator'] = model.netg.state_dict()
            model_result[seed_index]['Discriminator'] = model.netd.state_dict()
        elif opt.model == "BeatGAN":
            seed_index = "SEED" + str(seed)
            model_result[seed_index] = {}
            model_result[seed_index]['Generator'] = model.G.state_dict()
            model_result[seed_index]['Discriminator'] = model.D.state_dict()
        else:
            seed_index = "SEED" + str(seed)
            model_result[seed_index] = model.G.state_dict()

    chk_dir = '{}/{}'.format(results_dir, "model")
    if not os.path.exists(chk_dir):
        os.makedirs(chk_dir)

    torch.save(model_result, chk_dir + '/model_' + opt.model + '.pth')

    MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
    AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
    AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
    APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
    APs_seed_std = round(np.std(list(APs_seed.values())), 4)

    pres_seed_mean = round(np.mean(pres_seed), 4)
    pres_seed_std = round(np.std(pres_seed), 4)
    recall_seed_mean = round(np.mean(recall_seed), 4)
    recall_seed_std = round(np.std(recall_seed), 4)
    # f1_seed_mean = round(np.mean(f1_seed), 4)
    # f1_seed_std = round(np.std(f1_seed), 4)

    print("AUCs={}+{} \t APs={}+{} \t MAX_EPOCHs={} \t Pres={}+{} \t Recalls={}+{}".format(
        AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std, MAX_EPOCHs_seed, pres_seed_mean, pres_seed_std,
        recall_seed_mean, recall_seed_std))

    print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
        opt.model, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std, pres_seed_mean, pres_seed_std,
        recall_seed_mean, recall_seed_std,
        MAX_EPOCHs_seed_max
    ), file=file2print_detail)
    file2print_detail.flush()

    file2print.close()
    file2print_detail.close()
    # break

# if __name__ == '__main__':
#
#     results_dir='./log111'
#
#
#     opt.outf = results_dir
#     if not os.path.exists(results_dir):
#         os.makedirs(results_dir)
#
#     if opt.model == 'AE_CNN_noisy_multi':
#
#         file2print = open('{}/results_{}_{}-{}.log'.format(results_dir, opt.model, opt.NoisyType, opt.Snr), 'a+')
#         file2print_detail = open('{}/results_{}_{}-{}_detail.log'.format(results_dir, opt.model, opt.NoisyType, opt.Snr), 'a+')
#
#     else:
#
#         file2print = open('{}/results_{}.log'.format(results_dir, opt.model), 'a+')
#         file2print_detail = open('{}/results_{}_detail.log'.format(results_dir, opt.model), 'a+')
#
#     import datetime
#     print(datetime.datetime.now())
#     print(datetime.datetime.now(), file=file2print)
#     print(datetime.datetime.now(), file=file2print_detail)
#
#     if opt.model == 'AE_CNN_noisy_multi':
#
#         print('NoisyType is {}\t and SNR is{}\t'.format(opt.NoisyType, opt.Snr))
#         print('NoisyType is {}\t and SNR is{}\t'.format(opt.NoisyType, opt.Snr),file= file2print_detail)
#         print('NoisyType is {}\t and SNR is{}\t'.format(opt.NoisyType, opt.Snr),file= file2print)
#
#     print("Model\tDataset\tNormal_Label\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print_detail)
#
#     print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch")
#     print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std\tMax_Epoch", file=file2print)
#     file2print.flush()
#     file2print_detail.flush()
#
#     for dataset_name in list(DATASETS_NAME.keys()):
#         if dataset_name=='CWRU':
#             break
#         # dataset_name="CSPC"
#         # 噪声种类
#         # opt.plt_show=True
#         opt.noisy_classify = 6
#         if dataset_name=='CSPC':
#             opt.nc = 2
#         else:
#             opt.nc = 1
#         AUCs={}
#         APs={}
#         MAX_EPOCHs = {}
#         error=1
#
#         for normal_idx in range(DATASETS_NAME[dataset_name]):
#
#             print("[INFO] Dataset={}, Normal Label={}".format(dataset_name, normal_idx))
#             for i in range(10):
#                 opt.Snr=i+10
#                 MAX_EPOCHs_seed = {}
#                 AUCs_seed = {}
#                 APs_seed = {}
#                 for seed in SEEDS:
#                     np.random.seed(seed)
#                     opt.seed = seed
#
#                     opt.normal_idx = normal_idx
#                     try:
#                         dataloader, opt.isize, opt.signal_length = load_data(opt,dataset_name)
#                     except:
#                         error=0
#                         break
#                     opt.dataset = dataset_name
#
#
#                     #print("[INFO] Class Distribution: {}".format(class_stat))
#
#                     model = ModelTrainer(opt, dataloader, device)
#
#                     opt.name = "%s/%s" % (opt.model, opt.dataset)
#                     expr_dir = os.path.join(opt.outf, opt.name, 'train')
#                     test_dir = os.path.join(opt.outf, opt.name, 'test')
#
#                     if not os.path.isdir(expr_dir):
#                         os.makedirs(expr_dir)
#                     if not os.path.isdir(test_dir):
#                         os.makedirs(test_dir)
#
#                     args = vars(opt)
#                     file_name = os.path.join(expr_dir, 'opt.txt')
#                     with open(file_name, 'wt') as opt_file:
#                         opt_file.write('------------ Options -------------\n')
#                         for k, v in sorted(args.items()):
#                             opt_file.write('%s: %s\n' % (str(k), str(v)))
#                         opt_file.write('-------------- End ----------------\n')
#
#                     print(opt)
#
#                     print("################", dataset_name, "##################")
#                     print("################  Train  ##################")
#                     ap_test, auc_test, epoch_max_point = model.train()
#                     print("SEED: {}\t{}\t{}\t{}".format(seed,auc_test,ap_test,epoch_max_point),file=file2print)
#                     file2print.flush()
#                     AUCs_seed[seed] = auc_test
#                     APs_seed[seed] = ap_test
#                     MAX_EPOCHs_seed[seed] = epoch_max_point
#
#                 # End For
#         if error==0:
#             continue
#         MAX_EPOCHs_seed_max = round(np.max(list(MAX_EPOCHs_seed.values())), 4)
#         AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
#         AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
#         APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
#         APs_seed_std = round(np.std(list(APs_seed.values())), 4)
#
#         print("Dataset: {} \t Normal Label: {} \t AUCs={}+{} \t APs={}+{} \t MAX_EPOCHs={}".format(
#             dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std, MAX_EPOCHs_seed))
#
#         print("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
#             opt.model, dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std,
#             MAX_EPOCHs_seed_max
#         ), file=file2print_detail)
#         file2print_detail.flush()
#
#         AUCs[normal_idx] = AUCs_seed_mean
#         APs[normal_idx] = APs_seed_mean
#         MAX_EPOCHs[normal_idx] = MAX_EPOCHs_seed_max
#
#         print("{}\t{}\tTest\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{}".format(
#             opt.model, dataset_name, np.mean(list(AUCs.values())), np.std(list(AUCs.values())),
#             np.mean(list(APs.values())), np.std(list(APs.values())), np.max(list(MAX_EPOCHs.values()))
#         ), file=file2print)
#         file2print.flush()
#
