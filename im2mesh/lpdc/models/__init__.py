import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from im2mesh.lpdc.models import decoder, displacement
from im2mesh.utils.torchdiffeq.torchdiffeq import odeint, odeint_adjoint

decoder_dict = {
    'simple_local': decoder.DecoderCBatchNorm,
}

velocity_field_dict = {
    'concat': displacement.DisplacementDecoder,
}


class LPDCNet(nn.Module):
    '''LPDCNet model class.

    It consists of a decoder and, depending on the respective settings, an
    encoder, a temporal encoder, and a vector field.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        encoder_temporal (nn.Module): temporal encoder network
        vector_field (nn.Module): vector field network
        ode_step_size (float): step size of ode solver
        use_adjoint (bool): whether to use the adjoint method for obtaining
            gradients
        rtol (float): relative tolerance for ode solver
        atol (float): absolute tolerance for ode solver
        ode_solver (str): ode solver method
        p0_z (dist): prior distribution
        device (device): PyTorch device
        input_type (str): type of input

    '''

    def __init__(
        self, decoder, encoder=None, 
            encoder_temporal=None, vector_field=None,
            ode_step_size=None, use_adjoint=False,
            rtol=0.001, atol=0.00001, ode_solver='dopri5', p0_z=None,
            device=None, input_type=None, **kwargs):
        super().__init__()
        if p0_z is None:
            p0_z = dist.Normal(torch.tensor([]), torch.tensor([]))

        self.device = device
        self.input_type = input_type

        self.decoder = decoder
        self.encoder = encoder
        self.vector_field = vector_field
        self.encoder_temporal = encoder_temporal

        self.p0_z = p0_z
        self.rtol = rtol
        self.atol = atol
        self.ode_solver = ode_solver

        if use_adjoint:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint

        self.ode_options = {}
        if ode_step_size:
            self.ode_options['step_size'] = ode_step_size

    def forward(self, p, time_val, inputs, sample=True):
        ''' Makes a forward pass through the network.

        Args:
            p (tensor): points tensor
            time_val (tensor): time values
            inputs (tensor): input tensor
            sample (bool): whether to sample
        '''
        batch_size = p.size(0)
        c_s, c_t = self.encode_inputs(inputs)

        p_t_at_t0 = self.model.transform_to_t0(time_val, p, c_t=c_t)
        out = self.model.decode(p_t_at_t0, c=c_s)
        return out

    def decode(self, p, z=None, c=None, **kwargs):
        ''' Returns occupancy values for the points p at time step 0.

        Args:
            p (tensor): points
            z (tensor): latent code z
            c (tensor): latent conditioned code c (For OFlow, this is
                c_spatial)
        '''
        logits = self.decoder(p, z, c, **kwargs)
        p_r = dist.Bernoulli(logits=logits)
        return p_r


    def infer_z(self, inputs, c=None, data=None):
        ''' Infers a latent code z.

        The inputs and latent conditioned code are passed to the latent encoder
        to obtain the predicted mean and standard deviation.

        Args:
            inputs (tensor): input tensor
            c (tensor): latent conditioned code c
        '''
        batch_size = inputs.size(0)
        mean_z = torch.empty(batch_size, 0).to(self.device)
        logstd_z = torch.empty(batch_size, 0).to(self.device)

        q_z = dist.Normal(mean_z, torch.exp(logstd_z))
        q_z_t = dist.Normal(mean_z, torch.exp(logstd_z))

        return q_z, q_z_t

    def get_z_from_prior(self, size=torch.Size([]), sample=True):
        ''' Returns z from the prior distribution.

        If sample is true, z is sampled, otherwise the mean is returned.

        Args:
            size (torch.Size): size of z
            sample (bool): whether to sample z
        '''
        if sample:
            z_t = self.p0_z.sample(size).to(self.device)
            z = self.p0_z.sample(size).to(self.device)
        else:
            z = self.p0_z.mean.to(self.device)
            z = z.expand(*size, *z.size())
            z_t = z

        return z, z_t

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    # ######################################################
    # #### Encoding related functions #### #
    def encode_inputs(self, inputs):
        ''' Returns spatial and temporal latent code for inputs.

        Args:
            inputs (tensor): inputs tensor
        '''

        #spatiotemporal
        batch_size = inputs.shape[0]
        device = self.device

        if self.encoder_temporal is not None:
            c_s, c_t = self.encoder_temporal(inputs)
        else:
            #change size ?
            c_s = torch.empty(batch_size, 0).to(device)
            c_t = torch.empty(batch_size, 0).to(device)
        return c_s, c_t

    # ######################################################
    # #### Forward and Backward Flow functions #### #

    def transform_to_t(self, t, p, c_t=None):
        '''  Transforms points p from time 0 to (multiple) time values t.

        Args:
            t (tensor): time values
            p (tensor); points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t

        '''
        p_out, _ = self.eval_ODE(t, p, c_t=c_t, return_start=(0 in t))

        return p_out

    def transform_to_t0(self, t, p, c_t=None):
        ''' Transforms the points p at time t to time 0.

        Args:
            t (tensor): time values of the points
            p (tensor): points tensor
            z (tensor): latent code z
            c_t (tensor): latent conditioned temporal code c_t
        '''

        p_out, t_order = self.eval_ODE(t, p, c_t=c_t, t_batch=t,
                                       invert=True, return_start=True)

        # Select respective time value for each item from batch
        #batch_size = len(t_order)
        #p_out = p_out[torch.arange(batch_size), t_order]
        return p_out

    # ######################################################
    # #### ODE related functions and helper functions #### #

    def eval_ODE(self, t, p, c_t=None, t_batch=None, invert=False, return_start=False):
        c_dim = c_t.shape[-1]

        if invert:
            t_cur_eval = t
            t_delate_eval = torch.zeros(1).to(p.device)
        else:
            t_cur_eval = torch.zeros(1).to(p.device)
            t_delate_eval = t

        p, t_cur, t_delta = self.concat_vf_input(p, t_cur_eval, t_delate_eval)
        p_out = self.vector_field(p, t_cur, t_delta, c_t)
        return p_out, None

    def return_time_steps(self, t):
        ''' Returns time steps for the ODE Solver.
        The time steps are ordered, duplicates are removed, and time 0
        is added for the start.

        Args:
            t (tensor): time values
        '''
        device = self.device
        t_steps_eval, t_order = torch.unique(
            torch.cat([torch.zeros(1).to(device), t]), sorted=True,
            return_inverse=True)
        return t_steps_eval, t_order[1:]

    def disentangle_vf_output(self, v_out, p_dim=3, c_dim=None, return_start=False):
        ''' Disentangles the output of the velocity field.

        The inputs and outputs for / of the velocity network are concatenated
        to be able to use the adjoint method.

        Args:
            v_out (tensor): output of the velocity field
            p_dim (int): points dimension
            c_dim (int): dimension of conditioned code c
            z_dim (int): dimension of latent code z
            return_start (bool): whether to return start points
        '''

        batch_size, n_steps, _, _ = v_out.shape

        if c_dim is not None and c_dim != 0:
            v_out = v_out[:, :, :, :-c_dim]

        if not return_start:
            v_out = v_out[1:]
        return v_out

    def concat_vf_input(self, p, t_cur_eval, t_delta_eval, c=None):
        ''' Concatenate points p and latent code c to use it as input for ODE Solver.

        p of size (B x T x dim) and c of size (B x c_dim) and z of size
        (B x z_dim) is concatenated to obtain a tensor of size
        (B x (T*dim) + c_dim + z_dim).

        This is done to be able to use to the adjont method for obtaining
        gradients.

        Args:
            p (tensor): points tensor
            c (tensor): latent conditioned code c
            c (tensor): latent code z
        '''
        n_steps = len(t_delta_eval)
        batch_size, npoints, dim = p.shape
        t_cur = t_cur_eval[:, None].expand(batch_size, 1).to(p.device)
        t_delta = t_delta_eval[:, None].expand(batch_size, 1).to(p.device)
        return p, t_cur, t_delta
