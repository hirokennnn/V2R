import math
import copy
import os.path

import numpy as np
import torch
import VL_class_files
from VL_keep_pot import calc_keep_pot_pgrad
from VL_keep_pot import calc_keeppot
from VL_keep_pyr_pot import calc_keep_pyr_pot_pgrad
from VL_keep_pyr_pot import calc_keep_pyr_pot
from VL_LJ_asym_ell_pot import calc_LJ_asym_ell_pot_pgrad
from VL_LJ_asym_ell_pot import conbine_xyz_phi
from VL_LJ_asym_ell_pot import calc_ene
from tools_unit_constant import unit_ang2au
from tools_unit_constant import unit_kcal2hartree
from tools_unit_constant import unit_deg2rad

def find_param_tag(info_tag):
    dat_info_tag = {}
    for iligand in info_tag:
        for ipot in info_tag[iligand]:
            for iterm in info_tag[iligand][ipot]:
                if '@@' in str(info_tag[iligand][ipot][iterm]):
                    param_tag_tmp = info_tag[iligand][ipot][iterm]
                    if param_tag_tmp not in dat_info_tag.keys():
                        dat_info_tag[param_tag_tmp] = []
                    dat_info_tag[param_tag_tmp].append([iligand, ipot, iterm])
    return dat_info_tag

def set_unit(param_tag):
    list_length = ['a1', 'a2', 'b1', 'b2', 'c1', 'c2', 'dist', 'r0']
    list_angle = ['ang', 'phi']
    list_ene = ['eps']
    list_nounit = ['n_val', 'k_val']

    dat_convert = {}
    dat_convert['energy'] = unit_kcal2hartree()
    dat_convert['length'] = unit_ang2au()
    dat_convert['angle'] = unit_deg2rad()
    dat_convert['nounit'] = 1.0


    param_dim = []
    for iparam in list_length:
        if iparam in param_tag:
            param_dim.append('length')
    for iparam in list_ene:
        if iparam in param_tag:
            param_dim.append('energy')
    for iparam in list_angle:
        if iparam in param_tag:
            param_dim.append('angle')
    for iparam in list_nounit:
        if iparam in param_tag:
            param_dim.append('nounit')

    param_dim = list(set(param_dim))
    if len(param_dim) != 1:
        print(f'[ERROR] cannot define dimension : {param_tag}')
        print('exit...')
        exit()
    else:
        return dat_convert[param_dim[0]]

def VL_param_grad(xyz, param, param_tag, MM_param_list, path_phi_log):
    dat_param_grad = {}

    ###-----
    # calculate grad for keep pot (the keep potential)
    ###-----
    dat_keep_info_tag = find_param_tag(param_tag.keep_info)

    tensor_list = []
    tensor_tag = []
    for itag in dat_keep_info_tag:
        term_0 = dat_keep_info_tag[itag][0]
        tensor_list.append(torch.tensor(param.keep_info[term_0[0]][term_0[1]][term_0[2]], requires_grad=False))
        tensor_tag.append(itag)

    if len(tensor_tag) >= 1:
        grad_keeppot = torch.func.jacrev(calc_keep_pot_pgrad, argnums=2)(xyz, param_tag.keep_info, tensor_list, tensor_tag)
        for itag in range(len(tensor_tag)):
            dat_param_grad[tensor_tag[itag]] = grad_keeppot[itag].item()

    ###-----
    # calculate grad for keep pyr pot (the keep angle potential)
    ###-----
    dat_keeppyr_info_tag = find_param_tag(param_tag.keeppyr_info)

    tensor_list = []
    tensor_tag = []
    for itag in dat_keeppyr_info_tag:
        term_0 = dat_keeppyr_info_tag[itag][0]
        tensor_list.append(torch.tensor(param.keeppyr_info[term_0[0]][term_0[1]][term_0[2]], requires_grad=False))
        tensor_tag.append(itag)

    if len(tensor_tag) >= 1:
        grad_keeppyrpot = torch.func.jacrev(calc_keep_pyr_pot_pgrad, argnums=2)(xyz, param_tag.keeppyr_info, tensor_list, tensor_tag)
        for itag in range(len(tensor_tag)):
            dat_param_grad[tensor_tag[itag]] = grad_keeppyrpot[itag].item()


    ###-----
    # calculate grad for LJpot_asym_ell (the ovoid LJ potential)
    ###-----
    dat_LJ_asym_ell_info_tag = find_param_tag(param_tag.LJ_asym_ell_info)

    tensor_list = []
    tensor_tag = []
    for itag in dat_LJ_asym_ell_info_tag:
        term_0 = dat_LJ_asym_ell_info_tag[itag][0]
        tensor_list.append(torch.tensor(param.LJ_asym_ell_info[term_0[0]][term_0[1]][term_0[2]], requires_grad=False))
        tensor_tag.append(itag)

    if len(tensor_tag) >= 1:
        phi_log = VL_class_files.PhiLog(path_phi_log)
        phi_list = torch.tensor(phi_log.phi_list, requires_grad=False, dtype=torch.float64)
        grad_LJ_pot_asym_ell = torch.func.jacrev(calc_LJ_asym_ell_pot_pgrad, argnums=4)(xyz, phi_list, MM_param_list, param_tag.LJ_asym_ell_info, tensor_list, tensor_tag)
        for itag in range(len(tensor_tag)):
            dat_param_grad[tensor_tag[itag]] = grad_LJ_pot_asym_ell[itag].item()
        del grad_LJ_pot_asym_ell

    with open(param.path + '_grad', mode='w') as g:
        for itag in dat_param_grad:
            g.write('{0:<15} {1}\n'.format(itag, dat_param_grad[itag] * set_unit(itag)))


def calc_tot_ene(xyz, keep_info, keeppyr_info, LJ_asym_ell_info, MM_param_list, phi_list):
    keeppot_ene = calc_keeppot(xyz, keep_info)
    keeppyr_ene = calc_keep_pyr_pot(xyz, keeppyr_info)

    xyz_phi_tensor = conbine_xyz_phi(xyz, phi_list)
    ovoidLJ_ene = calc_ene(xyz_phi_tensor, MM_param_list, LJ_asym_ell_info)
    tot_ene = keeppot_ene + keeppyr_ene + ovoidLJ_ene
    return tot_ene


def calc_all_penarty(xyz, phi_list, MM_param_list, param_tag, tensor_list, tensor_tag, tensor_flag):
    keep_tensor_list, keep_tensor_tag = tensor_list[:tensor_flag[0]], tensor_tag[:tensor_flag[0]]
    keep_info = copy.deepcopy(param_tag.keep_info)
    for itag in range(len(keep_tensor_tag)):
        for iligand in keep_info:
            for ipot in keep_info[iligand]:
                for iterm in keep_info[iligand][ipot]:
                    if keep_info[iligand][ipot][iterm] == keep_tensor_tag[itag]:
                        keep_info[iligand][ipot][iterm] = keep_tensor_list[itag]

    keeppyr_tensor_list, keeppyr_tensor_tag = tensor_list[tensor_flag[0]:tensor_flag[1]], tensor_tag[tensor_flag[0]:tensor_flag[1]]
    keeppyr_info = copy.deepcopy(param_tag.keeppyr_info)
    for itag in range(len(keeppyr_tensor_tag)):
        for iligand in keeppyr_info:
            for ipot in keeppyr_info[iligand]:
                for iterm in keeppyr_info[iligand][ipot]:
                    if keeppyr_info[iligand][ipot][iterm] == keeppyr_tensor_tag[itag]:
                        keeppyr_info[iligand][ipot][iterm] = keeppyr_tensor_list[itag]

    LJ_tensor_list, LJ_tensor_tag = tensor_list[tensor_flag[1]:tensor_flag[2]], tensor_tag[tensor_flag[1]:tensor_flag[2]]
    LJ_asym_ell_info = copy.deepcopy(param_tag.LJ_asym_ell_info)
    for itag in range(len(LJ_tensor_tag)):
        for iligand in LJ_asym_ell_info:
            for ipot in LJ_asym_ell_info[iligand]:
                for iterm in LJ_asym_ell_info[iligand][ipot]:
                    if LJ_asym_ell_info[iligand][ipot][iterm] == LJ_tensor_tag[itag]:
                        LJ_asym_ell_info[iligand][ipot][iterm] = LJ_tensor_list[itag]

    tot_ene = calc_tot_ene(xyz, keep_info, keeppyr_info, LJ_asym_ell_info, MM_param_list, phi_list)
    return tot_ene

def VL_param_hess(xyz, param, param_tag, MM_param_list, path_phi_log, linkjob, comfile):
    if not os.path.isfile(path_phi_log):
        with open(path_phi_log, mode='w') as f:
            f.write("\n-----\n\n")
    phi_log = VL_class_files.PhiLog(path_phi_log)
    phi_list = torch.tensor(phi_log.phi_list, requires_grad=False, dtype=torch.float64)



    flag_num = 0
    tensor_list = []
    tensor_tag = []
    dat_keep_info_tag = find_param_tag(param_tag.keep_info)
    for itag in dat_keep_info_tag:
        term_0 = dat_keep_info_tag[itag][0]
        tensor_list.append(torch.tensor(param.keep_info[term_0[0]][term_0[1]][term_0[2]], requires_grad=False))
        tensor_tag.append(itag)
        flag_num += 1
    flag_keeppot = flag_num

    dat_keeppyr_info_tag = find_param_tag(param_tag.keeppyr_info)
    for itag in dat_keeppyr_info_tag:
        term_0 = dat_keeppyr_info_tag[itag][0]
        tensor_list.append(torch.tensor(param.keeppyr_info[term_0[0]][term_0[1]][term_0[2]], requires_grad=False))
        tensor_tag.append(itag)
        flag_num += 1
    flag_keeppyr = flag_num

    dat_LJ_asym_ell_info_tag = find_param_tag(param_tag.LJ_asym_ell_info)
    for itag in dat_LJ_asym_ell_info_tag:
        term_0 = dat_LJ_asym_ell_info_tag[itag][0]
        tensor_list.append(torch.tensor(param.LJ_asym_ell_info[term_0[0]][term_0[1]][term_0[2]], requires_grad=False))
        tensor_tag.append(itag)
        flag_num += 1
    flag_ovoidLJ = flag_num
    tensor_flag = [flag_keeppot, flag_keeppyr, flag_ovoidLJ]
    tensor_list = torch.tensor(tensor_list, dtype=torch.float64)

    ddV_vl_dqdq =  torch.func.jacrev(torch.func.jacrev(calc_all_penarty, argnums=1), argnums=1)(xyz, phi_list, MM_param_list, param_tag, tensor_list, tensor_tag, tensor_flag)

    # ddV_vl_dqdq_inv = torch.linalg.inv(ddV_vl_dqdq)
    ddV_vl_dqdq = 0.5 * (ddV_vl_dqdq + ddV_vl_dqdq.T)     # hessian の対称化
    eig_val_q, eig_vec_q = torch.linalg.eigh(ddV_vl_dqdq) # eig と比べ、対称行列で使うと高速になる。（三角行列のみ参照する。）

    eig_atol_q = 1.0e-12 # 絶対的に 10^−12 以下の小さな値or 最大固有値と比べて 10^−10 以下の小さな値をゼロと見なす固有値の閾値を設定
    eig_rtol_q = 1.0e-10
    eig_scale_q = torch.max(torch.abs(eig_val_q)).item()
    eig_tol_q = max(eig_atol_q, eig_rtol_q * eig_scale_q)

    stable_mask_q = torch.abs(eig_val_q) > eig_tol_q # ほぼゼロの固有値を除外
    eig_val_q_keep = eig_val_q[stable_mask_q]
    eig_vec_q_keep = eig_vec_q[:, stable_mask_q]
    ddV_vl_dqdq_inv = (eig_vec_q_keep / eig_val_q_keep.unsqueeze(0)) @ eig_vec_q_keep.T # Moore-Penrose疑似逆行列を構築(unsqueeze(0):行列の形を計算にある形に整える。)

    ddV_vl_dqdp = torch.func.jacrev(torch.func.jacrev(calc_all_penarty, argnums=1), argnums=4)(xyz, phi_list, MM_param_list, param_tag, tensor_list, tensor_tag, tensor_flag)
    dq_dp = -1 * ddV_vl_dqdq_inv @ ddV_vl_dqdp

    ddV_vl_dpdp = torch.func.jacrev(torch.func.jacrev(calc_all_penarty, argnums=4), argnums=4)(xyz, phi_list, MM_param_list, param_tag, tensor_list, tensor_tag, tensor_flag)

    ddV_vl_dqdQ_raw = torch.func.jacrev(torch.func.jacrev(calc_all_penarty, argnums=1), argnums=0)(xyz, phi_list, MM_param_list, param_tag, tensor_list, tensor_tag, tensor_flag)
    ddV_vl_dqdQ = ddV_vl_dqdQ_raw.reshape(len(phi_list), len(xyz) * 3)

    ddV_vl_dpdQ_raw = torch.func.jacrev(torch.func.jacrev(calc_all_penarty, argnums=4), argnums=0)(xyz, phi_list, MM_param_list, param_tag, tensor_list, tensor_tag, tensor_flag)
    ddV_vl_dpdQ = ddV_vl_dpdQ_raw.reshape(len(tensor_list), len(xyz) * 3)

    ddE_dQdQ_tril = np.zeros((len(xyz) * 3, len(xyz) * 3))
    for iline in range(len(linkjob.hess_tril)):
        for jterm in range(len(linkjob.hess_tril[iline])):
            ddE_dQdQ_tril[iline][jterm] = float(linkjob.hess_tril[iline][jterm])
    ddE_dQdQ = ddE_dQdQ_tril + ddE_dQdQ_tril.T - np.diag(ddE_dQdQ_tril) * np.eye(len(xyz) * 3)
    ddE_dQdQ = torch.tensor(ddE_dQdQ, dtype=torch.float64)

    dq_dQ = -1 * ddV_vl_dqdq_inv @ ddV_vl_dqdQ
    tmp_mat_1 = ddE_dQdQ + ddV_vl_dqdQ.T @ dq_dQ
    tmp_mat_1 = 0.5 * (tmp_mat_1 + tmp_mat_1.T) # 対称行列にする
    tmp_mat_2 = ddV_vl_dpdQ.T + ddV_vl_dqdQ.T @ dq_dp

    # eig_val, eig_vec = torch.linalg.eig(tmp_mat_1)
    # eig_val = eig_val[:-6]
    # eig_mat_inv = torch.eye((len(eig_val))) / eig_val
    # eig_vec = eig_vec.T[:-6].T
    """
    Frozen原子がある場合、残る剛体回転の数をFrozen原子の配置から判定する。
       固定なし                 ：6 つの自由度を除く。
       1原子固定                ：固定原子を中心とする回転3自由度を除く。
       2原子固定 or 3原子が直線状：固定原子の配置が同一直線上なら、その軸まわりの回転1自由度のみを除く。
       3原子以上固定            ：固有値を除かない。
    """
    eig_val, eig_vec = torch.linalg.eigh(tmp_mat_1) # 対称行列なら、eigh が適切.

    # 剛体モードの数を判定
    if comfile.tag_frozen:
        frozen_coord = np.array([atom_info[1] for atom_info in comfile.frozen_xyz], dtype=float)
        if len(frozen_coord) == 1:
            n_rigid_mode = 3
        else:
            frozen_relative = frozen_coord[1:] - frozen_coord[0]
            frozen_rank = np.linalg.matrix_rank(frozen_relative, tol=1.0e-10)
            if frozen_rank == 0:
                n_rigid_mode = 3
            elif frozen_rank == 1:
                n_rigid_mode = 1
            else:
                n_rigid_mode = 0
    else:
        n_rigid_mode = 6

    keep_mask = torch.ones_like(eig_val, dtype=torch.bool) # eig_valと同じ形状を持ち、すべての要素がTrue

    # 既知の並進・回転モードを除く
    if n_rigid_mode > 0:
        remove_idx = torch.argsort(torch.abs(eig_val))[:n_rigid_mode]
        keep_mask[remove_idx] = False
        removed_eig_val = eig_val[remove_idx].detach().cpu().numpy()
    
    # 剛体モード以外でも、閾値以下の固有値は疑似逆行列で0として扱う -> Moore-Penrose疑似逆行列として安定
    eig_atol = 1.0e-12
    eig_rtol = 1.0e-10
    eig_scale = torch.max(torch.abs(eig_val)).item()
    eig_tol = max(eig_atol, eig_rtol * eig_scale)
    stable_mask = keep_mask & (torch.abs(eig_val) > eig_tol)
    eig_val_keep = eig_val[stable_mask]
    eig_vec_keep = eig_vec[:, stable_mask]
    tmp_mat_1_inv = (eig_vec_keep / eig_val_keep.unsqueeze(0)) @ eig_vec_keep.T

    # tmp_mat_1_inv = (eig_vec @ eig_mat_inv @ eig_vec.T).double()
    dQ_dp = -1 * tmp_mat_1_inv @ tmp_mat_2
    ddE_dpdp = ddV_vl_dpdQ @ dQ_dp + ddV_vl_dpdp + ddV_vl_dqdp.T @ (dq_dQ @ dQ_dp + dq_dp)

    ddE_dpdp = ddE_dpdp.numpy()

    mat_unit_conv = []
    for iparam in tensor_tag:
        irow = []
        for jparam in tensor_tag:
            irow.append(set_unit(iparam) * set_unit(jparam))
        mat_unit_conv.append(irow)
    mat_unit_conv = np.array(mat_unit_conv)
    ddE_dpdp_unit = ddE_dpdp * mat_unit_conv
    np.savetxt(param.path + '_hess', ddE_dpdp_unit, delimiter=",")



