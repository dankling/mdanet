import os
from net import Mnet
import torch
import random
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from lib.dataset import Data
from lib.datapreloader import DataPrefetcher
from torch.nn import functional as F

from discriminator import FCDiscriminator
from cm import Curriculum
from tqdm import tqdm
import ot
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def compute_task_weights(M1: torch.Tensor,
                                M2: torch.Tensor,
                                losses: torch.Tensor,
                                epsilon: float = 0.05,
                                sinkhorn_iters: int = 50) -> torch.Tensor:
    """
    Compute dynamic task weights ω via entropic Sinkhorn OT.
    M1, M2: [H,W] confidence maps
    losses: [3,] tensor of L1, L2, L3
    returns ω: [3,] sum=1, on CUDA
    """
    # 1) initial distribution p0 from M1, M2 means + baseline
    r1 = float(M1.mean().item())
    r2 = float(M2.mean().item())
    r3 = 1.0
    denom = r1 + r2 + r3 + 1e-12
    p0 = np.array([r1/denom, r2/denom, r3/denom], dtype=np.float64)

    # 2) uniform target q
    q = np.array([1/3, 1/3, 1/3], dtype=np.float64)

    # 3) cost matrix C_ij = (L_i - L_j)^2
    L = losses.detach().cpu().numpy().astype(np.float64)
    C = np.zeros((3,3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if i != j:
                C[i,j] = (L[i] - L[j])**2

    # 4) solve Sinkhorn
    Gamma = ot.sinkhorn(p0, q, C, reg=epsilon, numItermax=sinkhorn_iters)

    # 5) ω_i = sum_j Γ_{i,j}
    omega = torch.from_numpy(Gamma.sum(axis=1)).float().cuda()
    return omega

def loss_1(score1, score2, score3, score4, label):

    score1 = F.interpolate(score1, label.shape[2:], mode='bilinear', align_corners=True)
    score2 = F.interpolate(score2, label.shape[2:], mode='bilinear', align_corners=True)
    score3 = F.interpolate(score3, label.shape[2:], mode='bilinear', align_corners=True)
    score4 = F.interpolate(score4, label.shape[2:], mode='bilinear', align_corners=True)


    sal_loss1 = F.binary_cross_entropy_with_logits(score1, label, reduction='mean')
    sal_loss2 = F.binary_cross_entropy_with_logits(score2, label, reduction='mean')
    sal_loss3 = F.binary_cross_entropy_with_logits(score3, label, reduction='mean')
    sal_loss4 = F.binary_cross_entropy_with_logits(score4, label, reduction='mean')

    return sal_loss1 + sal_loss2 + sal_loss3 + sal_loss4


if __name__ == '__main__':
    random.seed(118)
    np.random.seed(118)
    torch.manual_seed(118)
    torch.cuda.manual_seed(118)
    torch.cuda.manual_seed_all(118)
    # dataset

    source1_root = 'datasets/MTL_AD/ostrich1/'
    source2_root = 'datasets/MTL_AD/ostrich1/'
    target_root = 'datasets/logo/'
    source3_root = 'datasets/MTL_AD/ostrich1/'
    source4_root = 'datasets/MTL_AD/ostrich1/'

    save_path = './model'                                                                                                                                                                                                         
    if not os.path.exists(save_path): os.mkdir(save_path)
    lr = 0.001
    lr_d = 0.001
    lr_cm1 = 0.0001
    lr_cm2 = 0.0001
    lr_cm3 = 0.0001 
    lambda_adv_target = 0.01
    second_lambda_adv_target = 0.01
    third_lambda_adv_target = 0.01
    batch_size = 4
    epoch = 20
     
    lr_dec = [130, 145]

    #   source1:0 ; source2:1;  source313: 2; NET2 :3 ;target:4; TILE:5

    source1_data = Data(source1_root, 0)
    source2_data = Data(source2_root, 1)
    source3_data = Data(source3_root, 2)
    source4_data = Data(source4_root, 3)
    target_data = Data(target_root, 4)
    
   
    #  init net
    net = Mnet().cuda()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005, momentum=0.9)
    net.train()

    #  init dis
    model_D1 = FCDiscriminator(num_classes=3).cuda()  # 对应任务1
    model_D2 = FCDiscriminator(num_classes=3).cuda()  # 对应任务2
    model_D3 = FCDiscriminator(num_classes=5).cuda()  # 对应任务3
    model_D3.train()
    model_D3.cuda()
    model_D2.train()
    model_D2.cuda()
    model_D1.train()
    model_D1.cuda()
    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=lr_d, betas=(0.9, 0.99))
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=lr_d, betas=(0.9, 0.99))
    optimizer_D3 = optim.Adam(model_D3.parameters(), lr=lr_d, betas=(0.9, 0.99))
    bce_loss = torch.nn.CrossEntropyLoss()

    #  init cm
    weight_cm1 = Curriculum()
    optimizer_cm1 = optim.Adam(weight_cm1.parameters(), lr=lr_cm1, betas=(0.9, 0.99))
    weight_cm1.cuda()
    weight_cm1.train()
    weight_cm2 = Curriculum()
    optimizer_cm2 = optim.Adam(weight_cm2.parameters(), lr=lr_cm2, betas=(0.9, 0.99))
    weight_cm2.cuda()
    weight_cm2.train()
    weight_cm3 = Curriculum()
    optimizer_cm3 = optim.Adam(weight_cm2.parameters(), lr=lr_cm2, betas=(0.9, 0.99))
    weight_cm3.cuda()
    weight_cm3.train()


    num_params = 0
    for p in net.parameters():
        num_params += p.numel()
    print(num_params)

    for epochi in tqdm(range(1, epoch + 1), desc="Training Epochs"):
        if epochi in lr_dec:
            lr = lr / 10
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=0.0005,momentum=0.9)

            print(lr)
        

        domain_dataset_list = [source1_data, source2_data, target_data]

        domain_list = np.random.permutation(3)
        meta_train_domain_list = domain_list[:2]
        meta_test_domain_list = domain_list[2]
        meta_train_dataset = ConcatDataset([domain_dataset_list[meta_train_domain_list[0]],
                                            domain_dataset_list[meta_train_domain_list[1]],
                                            ])
        meta_train_loader = DataLoader(meta_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        prefetcher = DataPrefetcher(meta_train_loader)
        rgb, label, cla = prefetcher.next()
        print("cla:", cla)

        meta_test_dataset = domain_dataset_list[meta_test_domain_list]
        train_len = len(meta_train_dataset)
        test_len = len(meta_test_dataset)
        new_test_data = meta_test_dataset
        for k in range(train_len // test_len + 1):
            new_test_data = ConcatDataset([new_test_data, meta_test_dataset])
        meta_test_dataset_final = new_test_data
        meta_test_loader = DataLoader(meta_test_dataset_final, batch_size=batch_size, shuffle=True, num_workers=0)
        test_prefetcher = DataPrefetcher(meta_test_loader)
        test_rgb, test_label, test_cla = test_prefetcher.next()
        
        #下一个随即组。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。。
        #。
        #。
        #。
        #.
        
        second_domain_dataset_list = [source3_data, source4_data, target_data]

        second_domain_list = np.random.permutation(3)
        second_meta_train_domain_list = second_domain_list[:2]
        second_meta_test_domain_list = second_domain_list[2]
        second_meta_train_dataset = ConcatDataset([second_domain_dataset_list[second_meta_train_domain_list[0]],
                                                second_domain_dataset_list[second_meta_train_domain_list[1]],
                                                ])
        second_meta_train_loader = DataLoader(second_meta_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        second_prefetcher = DataPrefetcher(second_meta_train_loader)
        second_rgb, second_label, second_cla = second_prefetcher.next()
        print("second_cla:", second_cla)

        second_meta_test_dataset = second_domain_dataset_list[second_meta_test_domain_list]
        second_train_len = len(second_meta_train_dataset)
        second_test_len = len(second_meta_test_dataset)
        second_new_test_data = second_meta_test_dataset
        for k in range(second_train_len // second_test_len + 1):
            second_new_test_data = ConcatDataset([second_new_test_data, second_meta_test_dataset])
        second_meta_test_dataset_final = second_new_test_data
        second_meta_test_loader = DataLoader(second_meta_test_dataset_final, batch_size=batch_size, shuffle=True, num_workers=0)
        second_test_prefetcher = DataPrefetcher(second_meta_test_loader)
        second_test_rgb, second_test_label, second_test_cla = second_test_prefetcher.next()
        #第三个随即组
        
        third_domain_dataset_list = [source1_data, source2_data,source3_data, source4_data, target_data]

        third_domain_list = np.random.permutation(5)
        third_meta_train_domain_list = third_domain_list[:4]
        third_meta_test_domain_list = third_domain_list[4]
        third_meta_train_dataset = ConcatDataset([third_domain_dataset_list[third_meta_train_domain_list[0]],
                                                third_domain_dataset_list[third_meta_train_domain_list[1]],
                                                ])
        third_meta_train_loader = DataLoader(third_meta_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        third_prefetcher = DataPrefetcher(third_meta_train_loader)
        third_rgb, third_label, third_cla = third_prefetcher.next()

        third_meta_test_dataset = third_domain_dataset_list[third_meta_test_domain_list]
        third_train_len = len(third_meta_train_dataset)
        third_test_len = len(third_meta_test_dataset)
        third_new_test_data = third_meta_test_dataset
        for k in range(third_train_len // third_test_len + 1):
            third_new_test_data = ConcatDataset([third_new_test_data, third_meta_test_dataset])
        third_meta_test_dataset_final = third_new_test_data
        third_meta_test_loader = DataLoader(third_meta_test_dataset_final, batch_size=batch_size, shuffle=True, num_workers=0)
        third_test_prefetcher = DataPrefetcher(third_meta_test_loader)
        third_test_rgb, third_test_label, third_test_cla = third_test_prefetcher.next()

        print(train_len)
        print(test_len)
        print(len(meta_train_loader))
        print(len(meta_test_loader))
        print(second_train_len)
        print(second_test_len)
        print(len(second_meta_train_loader))
        print(len(second_meta_test_loader))
        print(third_train_len)
        print(third_test_len)
        
        print(len(third_meta_train_loader))
        print(len(third_meta_test_loader))
        s1 = net(rgb)[3]
        s2 = net(second_rgb)[3]
        s3 = net(third_rgb)[3]
        # 2) compute pure saliency losses
        L1 = loss_1(*net(rgb),   label)
        L2 = loss_1(*net(second_rgb), second_label)
        L3 = loss_1(*net(third_rgb),  third_label)
        # 3) compute confidence maps
        M1_map = torch.sigmoid(s1).mean(dim=0).squeeze(0).detach()
        M2_map = torch.sigmoid(s2).mean(dim=0).squeeze(0).detach()
        # 4) dynamic OT weight computation
        omega = compute_task_weights(M1_map, M2_map, torch.stack([L1, L2, L3]))
        

        iter_num = len(meta_train_loader)
        iter_num1 = len(second_meta_train_loader)
        iter_num2 = len(third_meta_train_loader)


        train_sal_loss = 0
        loss_adv_value = 0
        loss_D1_value = 0
        test_sal_loss = 0
        
        second_train_sal_loss = 0
        second_loss_adv_value = 0
        loss_D2_value = 0
        second_test_sal_loss = 0
        
        third_train_sal_loss = 0
        third_loss_adv_value = 0
        loss_D3_value = 0
        
        third_test_sal_loss = 0
        
        net.zero_grad()
        i = 0

#####task11111111111111111111111111




        with tqdm(range(iter_num), desc=f"Epoch {epochi} - Task 1 Progress") as pbar:
            for j in pbar:

                for param in model_D1.parameters():
                    param.requires_grad = False

                score1, score2, score3, score4 = net(rgb)
                # 在对抗训练的代码段之前，添加以下代码来打印 score4 的形状
                

                train_loss = omega[0] * loss_1(score1, score2, score3, score4, label)
                score_d1 = F.interpolate(score4, size=[352, 352], mode='bilinear', align_corners=True)
                
                train_sal_loss += train_loss.data
                train_loss.backward(retain_graph=True)
                w = weight_cm1(rgb)
                


                D1_out = model_D1(score_d1)
                label_mapping = {0: 0, 1: 1, 4: 2}
                cla_mapped = torch.tensor([label_mapping[int(c)] for c in cla]).cuda()

                # 将 cla_mapped 转换为张量并展开到目标尺寸
                cla_expanded = cla_mapped.view(-1, 1, 1).expand(-1, D1_out.shape[2], D1_out.shape[3]).long()
                loss_adv_train = bce_loss(D1_out, cla_expanded)
                

           
              

                loss_adv = lambda_adv_target * loss_adv_train
                loss_adv.backward(retain_graph=True)
                loss_adv_value += loss_adv.data
                

                for param in model_D1.parameters():
                    param.requires_grad = True

                score_d1 = score_d1.detach()
                D1_out = model_D1(score_d1)

                loss_D1_source = bce_loss(D1_out, cla_expanded.long())
                loss_D1 = w * loss_D1_source
                loss_D1.backward(loss_D1.clone().detach())
                loss_D1_value += loss_D1_source.data

                t_score1, t_score2, t_score3, t_score4, = net(test_rgb)
                test_loss = loss_1(t_score1, t_score2, t_score3, t_score4, test_label)
                test_sal_loss += test_loss.data
                test_loss.backward()
                optimizer.step()  # 更新网络权重
                optimizer.zero_grad()  # 清空累积的梯度

                optimizer_D1.step()  # 更新第一个判别器权重
                optimizer_D1.zero_grad()
                optimizer_cm1.step()  # 更新课程学习权重
                weight_cm1.zero_grad()
                pbar.set_postfix({
                "Train Loss": train_loss.item(),
                "Test Loss": test_loss.item(),
                "LAdv": loss_adv.item(),
                "Disc Loss": (loss_D1_value / iter_num).item()
                })


                rgb, label, cla = prefetcher.next()
                test_rgb, test_label, test_cla = test_prefetcher.next()
            
            
            
        #####task2222222
        
        
            
        with tqdm(range(iter_num1), desc=f"Epoch {epochi} - Task 2 Progress") as pbar:
            for j in pbar:
                for param in model_D2.parameters():
                    param.requires_grad = False

                second_score1, second_score2, second_score3, second_score4 = net(second_rgb)
                
                

                second_train_loss = omega[1] * loss_1(second_score1, second_score2, second_score3, second_score4, second_label)
                score_d2 = F.interpolate(second_score4, size=[352, 352], mode='bilinear', align_corners=True)
                
                second_train_sal_loss += second_train_loss.data
                second_train_loss.backward(retain_graph=True)
                w = weight_cm2(second_rgb)
                


                D2_out = model_D2(score_d2)
                second_label_mapping = {2: 0, 3: 1, 4: 2}
                
                # 使用 second_label_mapping 进行映射
                second_cla_mapped = torch.tensor([second_label_mapping[int(c)] for c in second_cla]).cuda()
                second_cla_expanded = second_cla_mapped.view(-1, 1, 1).expand(-1, D2_out.shape[2], D2_out.shape[3]).long()

                second_loss_adv_train = bce_loss(D2_out, second_cla_expanded)

                


                second_loss_adv = second_lambda_adv_target * second_loss_adv_train
                second_loss_adv.backward(retain_graph=True)
                second_loss_adv_value += second_loss_adv.data
                

                for param in model_D2.parameters():
                    param.requires_grad = True

                score_d2 = score_d2.detach()
                D2_out = model_D2(score_d2)

                loss_D2_source = bce_loss(D2_out, second_cla_expanded.long())
                
                loss_D2 = w * loss_D2_source
                loss_D2.backward(loss_D2.clone().detach())
                loss_D2_value += loss_D2_source.data
                

                second_t_score1, second_t_score2, second_t_score3, second_t_score4, = net(second_test_rgb)
                second_test_loss = loss_1(second_t_score1, second_t_score2, second_t_score3, second_t_score4, second_test_label)
                second_test_sal_loss += second_test_loss.data
                second_test_loss.backward()
                
                
                optimizer.step()  # 更新网络权重
                optimizer.zero_grad()  # 清空累积的梯度

            

                optimizer_D2.step()  # 更新第二个判别器权重
                optimizer_D2.zero_grad()

            

                optimizer_cm2.step()  # 更新第二个课程学习权重
                weight_cm2.zero_grad()
                pbar.set_postfix({
                "secondTrain Loss": train_loss.item(),
                "secondTest Loss": test_loss.item(),
                "secondLAdv": loss_adv.item(),
                "secondDisc Loss": (loss_D2_value / iter_num1).item()
                })

            
                second_rgb, second_label, second_cla = second_prefetcher.next()
                second_test_rgb, second_test_label, second_test_cla = second_test_prefetcher.next()

        #####task3333333333333333   
        
        
        
        with tqdm(range(iter_num2), desc=f"Epoch {epochi} - Task 3 Progress") as pbar:
            for j in pbar:
                for param in model_D3.parameters():
                    param.requires_grad = False

                third_score1, third_score2, third_score3, third_score4 = net(third_rgb)
                
                

                third_train_loss = omega[2] * loss_1(third_score1, third_score2, third_score3, third_score4, third_label)
                score_d3 = F.interpolate(third_score4, size=[352, 352], mode='bilinear', align_corners=True)
                
                third_train_sal_loss += third_train_loss.data
                third_train_loss.backward(retain_graph=True)
                w = weight_cm3(third_rgb)
                


                D3_out = model_D3(score_d3)
                
                third_cla_expanded = third_cla.view(-1, 1, 1).expand(-1, D3_out.shape[2], D3_out.shape[3])
                

                third_loss_adv_train = bce_loss(D3_out, third_cla_expanded.long())
                
                third_loss_adv = third_lambda_adv_target * third_loss_adv_train
                third_loss_adv.backward(retain_graph=True)
                third_loss_adv_value += third_loss_adv.data
                for param in model_D3.parameters():
                    param.requires_grad = True

                score_d3 = score_d3.detach()
                D3_out = model_D3(score_d3)
                loss_D3_source = bce_loss(D3_out, third_cla_expanded.long())
                loss_D3 = w * loss_D3_source
                loss_D3.backward(loss_D3.clone().detach())
                loss_D3_value += loss_D3_source.data

                third_t_score1, third_t_score2, third_t_score3, third_t_score4, = net(third_test_rgb)
                third_test_loss = loss_1(third_t_score1, third_t_score2, third_t_score3, third_t_score4, third_test_label)
                third_test_sal_loss += third_test_loss.data
                third_test_loss.backward()

            
                
                
                optimizer.step()  # 更新网络权重
                optimizer.zero_grad()  # 清空累积的梯度

            

                optimizer_D3.step()  # 更新第二个判别器权重
                optimizer_D3.zero_grad()

            

                optimizer_cm3.step()  # 更新第二个课程学习权重
                weight_cm3.zero_grad()
                pbar.set_postfix({
                "thirdTrain Loss": train_loss.item(),
                "thirdTest Loss": test_loss.item(),
                "thirdLAdv": loss_adv.item(),
                "thirdDisc Loss": (loss_D3_value / iter_num2).item()
                })
                third_rgb, third_label, third_cla = third_prefetcher.next()
                third_test_rgb, third_test_label, third_test_cla = third_test_prefetcher.next()
            
       

        
        
        if epochi >= 70 and epochi % 10 == 0:
            torch.save(net.state_dict(), '%s/epoch_%d.pth' % (save_path, epochi))
    torch.save(net.state_dict(), '%s/final.pth' % (save_path))
