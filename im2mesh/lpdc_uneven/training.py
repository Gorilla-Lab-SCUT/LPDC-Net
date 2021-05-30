import os
import torch
import numpy as np
from torch.nn import functional as F
from im2mesh.common import compute_iou
from torch import distributions as dist
from im2mesh.training import BaseTrainer
import time

class Trainer(BaseTrainer):
    ''' Trainer class for OFlow Model.

    Args:
        model (nn.Module): OFlow Model
        optimizer (optimizer): PyTorch optimizer
        device (device): PyTorch device
        input_type (str): input type
        vis_dir (str): visualization directory
        threshold (float): threshold value for ONet-based
            shape representation at time 0
        eval_sample (bool): whether to evaluate with sampling
            (for KL Divergence)
        loss_cor (bool): whether to train with correspondence loss
        loss_corr_bw (bool): whether to train correspondence loss
            also backwards
        loss_recon (bool): whether to train with reconstruction loss
        vae_beta (float): beta hyperparameter for VAE loss
    '''

    def __init__(self, model, optimizer, device=None, input_type='img',
                 vis_dir=None, threshold=0.3, eval_sample=False,
                 loss_corr=False, loss_corr_bw=False, loss_recon=True, vae_beta=0.0001, 
                 loss_transform_forward=False, length_sequence = 17, select_steps=[0, 16]):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        self.loss_corr = loss_corr
        self.loss_recon = loss_recon
        self.loss_corr_bw = loss_corr_bw
        self.vae_beta = vae_beta
        self.loss_transform_forward = loss_transform_forward
        self.length_sequence = length_sequence
        self.select_steps = select_steps

        # Check what metric to use for validation
        self.eval_iou = (self.model.decoder is not None and
                         self.model.vector_field is not None)

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        ''' Performs a train step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        device = self.device
        inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)
        eval_dict = {}
        loss = 0

        with torch.no_grad():
            # Encode inputs
            c_s, c_t = self.model.encode_inputs(inputs)

            # IoU
            if self.eval_iou:
                eval_dict_iou = self.eval_step_iou(data, c_s=c_s, c_t=c_t)
                for (k, v) in eval_dict_iou.items():
                    eval_dict[k] = v
                loss += eval_dict['rec_error']
            else:
                # Correspondence Loss
                eval_dict_mesh = self.eval_step_corr_l2(data, c_t=c_t)
                for (k, v) in eval_dict_mesh.items():
                    eval_dict[k] = v
                loss += eval_dict['l2']

        eval_dict['loss'] = loss
        return eval_dict

    def eval_step_iou(self, data, c_s=None, c_t=None):
        ''' Calculates the IoU score for an evaluation test set item.

        Args:
            data (dict): data dictionary
            c_s (tensor): spatial conditioned code
            c_t (tensor): temporal conditioned code
            z (tensor): latent shape code
            z_t (tensor): latent motion code
        '''
        device = self.device
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        eval_dict = {}

        pts_iou = data.get('points').to(device)
        occ_iou = data.get('points.occ').squeeze(0)
        pts_iou_t = data.get('points.time').to(device)

        batch_size, n_steps, n_pts, dim = pts_iou.shape
        ####unevn 
        index = torch.arange(n_steps).float().to(device)
        pts_iou_t = (index / float(n_steps-1))[None, :].expand(batch_size, n_steps)
        print('eval uneven', pts_iou.shape, pts_iou_t.shape)

        logits_pts_iou_nsteps = []
        for i in range(n_steps):
            pts_iou_at_t0 = self.model.transform_to_t0(pts_iou_t[:, i], pts_iou[:, i], c_t=c_t)
            logits_pts_iou_t = self.model.decode(pts_iou_at_t0, c=c_s[:, 0, :]).logits
            logits_pts_iou_nsteps.append(logits_pts_iou_t)
        logits_t0 = torch.stack(logits_pts_iou_nsteps, dim=1)

        # Calculate predicted occupancy values
        rec_error = F.binary_cross_entropy_with_logits(
            logits_t0.view(-1, n_pts), occ_iou.to(device).view(-1, n_pts),
            reduction='none')
        rec_error = rec_error.mean(-1)
        rec_error = rec_error.view(batch_size, n_steps).mean(0)

        occ_pred = (logits_t0 > threshold).view(
            batch_size, n_steps, n_pts).cpu().numpy()

        # Calculate IoU
        occ_gt = (occ_iou >= 0.5).numpy()
        iou = compute_iou(
            occ_pred.reshape(-1, n_pts), occ_gt.reshape(-1, n_pts))
        iou = iou.reshape(batch_size, n_steps).mean(0)

        eval_dict['iou'] = iou.sum() / len(iou)
        eval_dict['rec_error'] = rec_error.sum().item() / len(rec_error)
        for i in range(len(iou)):
            eval_dict['iou_t%d' % i] = iou[i]
            eval_dict['rec_error_t%d' % i] = rec_error[i].item()

        return eval_dict

    def eval_step_corr_l2(self, data, c_t=None):
        ''' Calculates the correspondence l2 distance for an evaluation test set item.

        Args:
            data (dict): data dictionary
            c_s (tensor): spatial conditioned code
            c_t (tensor): temporal conditioned code
            z (tensor): latent code
        '''
        eval_dict = {}
        device = self.device
        p_mesh = data.get('points_mesh').to(device)
        # t values are the same for every batch
        p_mesh_t = data.get('points_mesh.time').to(device)
        batch_size, n_steps, n_pts, p_dim = p_mesh.shape
        #unevn:
        index = torch.arange(n_steps).float().to(device)
        pts_iou_t = (index / float(n_steps-1))[None, :].expand(batch_size, n_steps)
        print('eval uneven', p_mesh.shape, pts_iou_t.shape)

        # Transform points on mesh from t=0 to later time steps
        #batch_size, n_steps, n_pts, p_dim = p_mesh.shape
        p_mesh_pred_batch = []
        for i in range(n_steps):
            p_mesh_pred = self.model.transform_to_t(p_mesh_t[:, i], p_mesh[:, 0], c_t=c_t)
            p_mesh_pred_batch.append(p_mesh_pred)
        pts_pred = torch.stack(p_mesh_pred_batch, dim=1)

        if self.loss_corr_bw:
            # Backwards prediction
            pred_b = self.model.transform_to_t_backward(p_mesh_t,
                                                        p_mesh[:, -1],
                                                        c_t).flip(1)

            # Linear Interpolate between both directions
            w = (torch.arange(n_steps).float() / (n_steps - 1)).view(
                1, n_steps, 1, 1).to(device)
            pts_pred = pts_pred * (1 - w) + pred_b * w

        # Calculate l2 distance between predicted and GT points
        l2 = torch.norm(pts_pred - p_mesh, 2, dim=-1).mean(0).mean(-1)

        eval_dict['l2'] = l2.sum().item() / len(l2)
        for i in range(len(l2)):
            eval_dict['l2_%d' % (i+1)] = l2[i].item()

        return eval_dict

    def visualize(self, data):
        ''' Visualizes visualization data.
        Currently not implemented!

        Args:
            data (tensor): visualization data dictionary
        '''
        print("Currently not implemented.")

    def get_loss_recon(self, data, c_s=None, c_t=None):
        ''' Computes the reconstruction loss.

        Args:
            data (dict): data dictionary
            c_s (tensor): spatial conditioned code
            c_t (tensor): temporal conditioned code
        '''
        if not self.loss_recon:
            return 0

        # loss_t0 = self.get_loss_recon_t_notransform(data, c_s)
        # loss_t_forward = self.get_loss_recon_t_forward(data, c_s, c_t)
        # return loss_t0 + loss_t_forward

        loss_t0 = self.get_loss_recon_t0(data, c_s)
        loss_t_forward = self.get_loss_recon_t_forward(data, c_s, c_t)
        loss_t_backward = self.get_loss_recon_t_backward(data, c_s, c_t)
        return loss_t0 + loss_t_forward + loss_t_backward

    ####################################
    def get_loss_recon_t0(self, data, c_s=None):
        device = self.device
        p_t0 = data.get('points').to(device)
        occ_t0 = data.get('points.occ').to(device)
        batch_size, n_pts, p_dim = p_t0.shape

        logits_p_t0 = self.model.decode(p_t0, c=c_s[:, 0]).logits
        loss_occ_t0 = F.binary_cross_entropy_with_logits(logits_p_t0.view(batch_size, -1), occ_t0.view(batch_size, -1), reduction='none')
        loss_occ_t0 = loss_occ_t0.mean()
        return loss_occ_t0
        
    def get_loss_recon_t_notransform(self, data, c_s=None):
        device = self.device
        p_t = data.get('points_t').to(device)
        occ_t = data.get('points_t.occ').to(device)
        time_val = data.get('points_t.time').to(device)
        batch_size, n_steps, n_pts, p_dim = p_t.shape

        logits_p_t_nsteps = []
        for i in range(n_steps):
            logits_p_t = self.model.decode(p_t[:, i, ...], c=c_s[:, i]).logits
            logits_p_t_nsteps.append(logits_p_t)
        logits_p_t_nsteps = torch.stack(logits_p_t_nsteps, dim=1)

        loss_occ_t = F.binary_cross_entropy_with_logits(logits_p_t_nsteps.view(batch_size, -1), \
        occ_t.view(batch_size, -1), reduction='none')
        loss_occ_t = loss_occ_t.mean()
        return loss_occ_t

    def get_loss_recon_t_backward(self, data, c_s=None, c_t=None):
        device = self.device

        p_t = data.get('points_t').to(device)
        occ_t = data.get('points_t.occ').to(device)
        time_val = data.get('points_t.time').to(device)
        batch_size, n_steps, n_pts, p_dim = p_t.shape
        #if unevn:
        index = torch.arange(n_steps).float().to(device)
        time_val = (index / float(n_steps-1))[None, :].expand(batch_size, n_steps)
        print('loss t uneven', p_t.shape, time_val.shape)

        logits_p_t_nsteps = []
        for i in range(n_steps):
            p_t_at_t0 = self.model.transform_to_t0(time_val[:, i], p_t[:, i], c_t=c_t)
            logits_p_t = self.model.decode(p_t_at_t0, c=c_s[:, 0]).logits
            logits_p_t_nsteps.append(logits_p_t)
        logits_p_t_nsteps = torch.stack(logits_p_t_nsteps, dim=1)

        loss_occ_t = F.binary_cross_entropy_with_logits(
            logits_p_t_nsteps.view(batch_size, -1), occ_t.view(batch_size, -1), reduction='none')
        loss_occ_t = loss_occ_t.mean()
        return loss_occ_t

    def get_loss_recon_t_forward(self, data, c_s=None, c_t=None):
        device = self.device

        p_t = data.get('points_t').to(device)
        occ_t = data.get('points_t.occ').to(device)
        time_val = data.get('points_t.time').to(device)
        batch_size, n_steps, n_pts, p_dim = p_t.shape
        #unevn:
        index = torch.arange(n_steps).float().to(device)
        time_val = (index / float(n_steps-1))[None, :].expand(batch_size, n_steps)
        print('loss t uneven', p_t.shape, time_val.shape)

        logits_p_t_nsteps = []
        for i in range(n_steps):
            p_t_at_t0 = self.model.transform_to_t(time_val[:, i], p_t[:, 0], c_t=c_t)
            logits_p_t = self.model.decode(p_t_at_t0, c=c_s[:, i]).logits
            logits_p_t_nsteps.append(logits_p_t)
        logits_p_t_nsteps = torch.stack(logits_p_t_nsteps, dim=1)

        loss_occ_t = F.binary_cross_entropy_with_logits(
            logits_p_t_nsteps.view(batch_size, -1), occ_t[:, 0:1, :].expand(batch_size, n_steps, n_pts).contiguous().view(batch_size, -1), reduction='none')
        loss_occ_t = loss_occ_t.mean()
        return loss_occ_t

    def compute_loss_corr(self, data, c_t=None):
        ''' Returns the correspondence loss.

        Args:
            data (dict): data dictionary
            c_t (tensor): temporal conditioned code c_s
            z_t (tensor): latent temporal code z
        '''
        if not self.loss_corr:
            return 0

        device = self.device
        # Load point cloud data which are provided in equally spaced time
        # steps between 0 and 1
        pc = data.get('pointcloud').to(device)
        #unevn:
        length_sequence = len(self.select_steps)
        t = (torch.arange(
            length_sequence, dtype=torch.float32) / (length_sequence - 1)
        ).to(device)
        print('loss corr uneven', t.shape)
        if self.loss_corr_bw:
            # Use forward and backward prediction
            pred_f = self.model.transform_to_t(t, pc[:, 0], c_t=c_t)

            pred_b = self.model.transform_to_t_backward(
                t, pc[:, -1], c_t=c_t)
            pred_b = pred_b.flip(1)

            lc1 = torch.norm(pred_f - pc, 2, dim=-1).mean()
            lc2 = torch.norm(pred_b - pc, 2, dim=-1).mean()
            loss_corr = lc1 + lc2
        else:
            batch_size, n_steps, n_pts, p_dim = pc.shape
            pc_pred_batch = []
            for i in range(n_steps):
                pc_pred = self.model.transform_to_t(t[None, i], pc[:, 0], c_t=c_t)
                pc_pred_batch.append(pc_pred)
            pc_pred_batch = torch.stack(pc_pred_batch, dim=1)
            loss_corr = torch.norm(pc_pred_batch - pc, 2, dim=-1).mean()
        return loss_corr

    def compute_loss(self, data):
        ''' Computes the loss.

        Args:
            data (dict): data dictionary
        '''
        t0 = time.time()
        device = self.device
        # Encode inputs
        inputs = data.get('inputs', torch.empty(1, 1, 0)).to(device)
        c_s, c_t = self.model.encode_inputs(inputs)
        t1 = time.time()

        # Losses
        # Reconstruction Loss
        loss_recon = self.get_loss_recon(data, c_s, c_t)
        t2 = time.time()

        # Correspondence Loss
        loss_corr = self.compute_loss_corr(data, c_t)
        t3 = time.time()

        loss = loss_recon + loss_corr
        print('total loss: %.4f, loss_recon: %.4f, loss_corr: %.4f'%(loss.item(), loss_recon.item(), loss_corr.item()))
        print('encode: %.4f, recon: %.4f, corr: %.4f'%(t1-t0, t2-t1, t3-t2))
        return loss
