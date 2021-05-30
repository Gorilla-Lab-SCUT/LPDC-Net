import torch
import torch.distributions as dist
from torch import nn
import os
from im2mesh.encoder import encoder_dict, encoder_temporal_dict
from im2mesh.lpdc import models, training, generation
from im2mesh import data

def get_decoder(cfg, device, dim=3, c_dim=0):
    ''' Returns a decoder instance.

    Args:  
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dim (int): points dimension
        c_dim (int): dimension of conditioned code c
        padding (float): the number of padding
    '''
    decoder = cfg['model']['decoder']
    decoder_kwargs = cfg['model']['decoder_kwargs']

    #positional encoding
    if 'pos_encoding' in cfg['model']:
        decoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    if decoder:
        decoder = models.decoder_dict[decoder](
            dim=dim, c_dim=c_dim, **decoder_kwargs).to(device)
    elif decoder == 'simple_local':
        decoder = models.decoder_dict[decoder](
            dim=dim,  c_dim=c_dim, **decoder_kwargs
        )
    else:
        decoder = None
    return decoder


def get_velocity_field(cfg, device, dim=3, c_dim=0):
    ''' Returns a velocity field instance.

    Args:  
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dim (int): points dimension
        c_dim (int): dimension of conditioned code c
        padding (float): the number of padding
    '''
    velocity_field = cfg['model']['velocity_field']
    velocity_field_kwargs = cfg['model']['velocity_field_kwargs']

    if velocity_field == 'concat':
        velocity_field = models.velocity_field_dict[velocity_field](
            out_dim=dim, c_dim=c_dim, **velocity_field_kwargs 
        ).to(device)
    else:
        velocity_field = None

    return velocity_field


def get_encoder(cfg, device, dataset=None, c_dim=0):
    ''' Returns an encoder instance.

    If input type if 'idx', the encoder consists of an embedding layer.

    Args:  
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        dataset (dataset): dataset
        c_dim (int): dimension of conditioned code c
    '''
    encoder = cfg['model']['encoder']
    encoder_kwargs = cfg['model']['encoder_kwargs']

    # local positional encoding
    if 'pos_encoding' in cfg['model']:
        encoder_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    if encoder == 'idx':
        if cfg['model']['learn_embedding']:
            encoder = nn.Sequential(
                nn.Embedding(len(dataset), 128),
                nn.Linear(128, c_dim)).to(device)
        else:
            encoder = nn.Embedding(len(dataset), c_dim).to(device)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            c_dim=c_dim, **encoder_kwargs).to(device)
    else:
        encoder = None

    return encoder

def get_encoder_temporal(cfg, device, dataset=None, c_dim=0):
    ''' Returns a temporal encoder instance.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    '''
    encoder_temporal = cfg['model']['encoder_temporal']
    encoder_temporal_kwargs = cfg['model']['encoder_temporal_kwargs']

    # local positional encoding
    if 'pos_encoding' in cfg['model']:
        encoder_temporal_kwargs['pos_encoding'] = cfg['model']['pos_encoding']

    if encoder_temporal:
        if encoder_temporal == 'idx':
            if cfg['model']['learn_embedding']:
                encoder_temporal = nn.Sequential(
                    nn.Embedding(len(dataset), 128),
                    nn.Linear(128, c_dim)).to(device)
            else:
                encoder_temporal = nn.Embedding(len(dataset), c_dim).to(device)
        else:
            encoder_temporal = encoder_temporal_dict[encoder_temporal](
             c_dim=c_dim, **encoder_temporal_kwargs).to(device)
    else:
        encoder_temporal = None

    return encoder_temporal


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Returns an OFlow model instance.

    Depending on the experimental setup, it consists of encoders,
    latent encoders, a velocity field, and a decoder instance.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
        c_dim (int): dimension of conditioned code c
        z_dim (int): dimension of latent code z
    '''
    # Shortcuts
    dim = cfg['data']['dim']
    z_dim = cfg['model']['z_dim']
    c_dim = cfg['model']['c_dim']
    input_type = cfg['data']['input_type']
    ode_solver = cfg['model']['ode_solver']
    ode_step_size = cfg['model']['ode_step_size']
    use_adjoint = cfg['model']['use_adjoint']
    rtol = cfg['model']['rtol']
    atol = cfg['model']['atol']

    # Get individual components
    decoder = get_decoder(cfg, device, dim, c_dim)
    velocity_field = get_velocity_field(cfg, device, dim, c_dim)
    encoder = get_encoder(cfg, device, dataset, c_dim)
    encoder_temporal = get_encoder_temporal(cfg, device, dataset, c_dim)
    p0_z = get_prior_z(cfg, device)

    model = models.LPDCNet(
        decoder=decoder, encoder=encoder, 
        encoder_temporal=encoder_temporal, vector_field=velocity_field,
        ode_step_size=ode_step_size, use_adjoint=use_adjoint,
        rtol=rtol, atol=atol, ode_solver=ode_solver,
        p0_z=p0_z, device=device, input_type=input_type)
    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns an OFlow trainer instance.

    Args:
        model (nn.Module): OFlow model
        optimizer (optimizer): PyTorch optimizer
        cfg (yaml config): yaml config object
        device (device): PyTorch device
    '''
    threshold = cfg['test']['threshold']
    out_dir = cfg['training']['out_dir']
    vis_dir = os.path.join(out_dir, 'vis')
    input_type = cfg['data']['input_type']
    loss_corr = cfg['model']['loss_corr']
    loss_recon = cfg['model']['loss_recon']
    loss_corr_bw = cfg['model']['loss_corr_bw']
    eval_sample = cfg['training']['eval_sample']
    vae_beta = cfg['model']['vae_beta']
    loss_transform_forward = cfg['model']['loss_transform_forward']
    #unevn = cfg['training']['unevn']
    #length_sequence = cfg['data']['length_sequence']
    #select_steps=cfg['data']['select_steps']

    trainer = training.Trainer(
        model, optimizer, device=device, input_type=input_type,
        vis_dir=vis_dir, threshold=threshold,
        eval_sample=eval_sample, loss_corr=loss_corr, 
        loss_recon=loss_recon, loss_corr_bw=loss_corr_bw,
        vae_beta=vae_beta, loss_transform_forward=loss_transform_forward)
    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns an OFlow generator instance.

    It provides methods to extract the final meshes from the OFlow
    representation.

    Args:
        model (nn.Module): OFlow model
        cfg (yaml config): yaml config object
        device (device): PyTorch device
    '''
    generator = generation.Generator3D(
        model,
        device=device,
        threshold=cfg['test']['threshold'],
        resolution0=cfg['generation']['resolution_0'],
        upsampling_steps=cfg['generation']['upsampling_steps'],
        padding=cfg['generation']['padding'],
        sample=cfg['generation']['use_sampling'],
        refinement_step=cfg['generation']['refinement_step'],
        simplify_nfaces=cfg['generation']['simplify_nfaces'],
        n_time_steps=cfg['generation']['n_time_steps'],
        mesh_color=cfg['generation']['mesh_color'],
        only_end_time_points=cfg['generation']['only_end_time_points'],
        interpolate=cfg['generation']['interpolate'],
        fix_z=cfg['generation']['fix_z'],
        fix_zt=cfg['generation']['fix_zt'],
        #select_steps=cfg['data']['select_steps'],
        #unevn = cfg['training']['unevn']，
    )

    return generator


def get_prior_z(cfg, device, **kwargs):
    ''' Returns the prior distribution of latent code z.

    Args:
        cfg (yaml config): yaml config object
        device (device): PyTorch device
    '''
    z_dim = cfg['model']['z_dim']
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def get_transforms(cfg):
    ''' Returns transform objects.

    Args:
        cfg (yaml config): yaml config object
    '''
    n_pcl = cfg['data']['n_training_pcl_points']
    n_pt = cfg['data']['n_training_points']
    n_pt_eval = cfg['training']['n_eval_points']

    transf_pt = data.SubsamplePoints(n_pt)
    transf_pt_back = data.SubsamplePointsSeq(n_pt, connected_samples=True, random=True)
    transf_pt_val = data.SubsamplePointsSeq(n_pt_eval, random=False)
    transf_pcl_val = data.SubsamplePointcloudSeq(n_pt_eval, random=False)
    transf_pcl = data.SubsamplePointcloudSeq(n_pcl, connected_samples=True)

    return transf_pt, transf_pt_back, transf_pt_val, transf_pcl, transf_pcl_val


def get_data_fields(mode, cfg):
    ''' Returns data fields.

    Args:
        mode (str): mode (train|val|test)
        cfg (yaml config): yaml config object
    '''
    fields = {}
    seq_len = cfg['data']['length_sequence']
    p_folder = cfg['data']['points_iou_seq_folder']
    pcl_folder = cfg['data']['pointcloud_seq_folder']
    mesh_folder = cfg['data']['mesh_seq_folder']
    generate_interpolate = cfg['generation']['interpolate']
    unpackbits = cfg['data']['points_unpackbits']
    select_steps= cfg['data']['select_steps']

    # Transformation
    transf_pt, transf_pt_back, transf_pt_val, transf_pcl, transf_pcl_val = get_transforms(cfg)

    # Fields
    pts_iou_field = data.PointsSubseqField
    pts_corr_field = data.PointCloudSubseqField

    if mode == 'train':
        if cfg['model']['loss_recon']:
            fields['points'] = pts_iou_field(p_folder, transform=transf_pt,
                                             seq_len=seq_len,
                                             fixed_time_step=0, 
                                             unpackbits=unpackbits)
            fields['points_t'] = pts_iou_field(p_folder, transform=transf_pt_back,
                                               all_steps=True, ####
                                               seq_len=seq_len, select_steps=select_steps,
                                               unpackbits=unpackbits)
        # Connectivity Loss:
        if cfg['model']['loss_corr']:
            fields['pointcloud'] = pts_corr_field(pcl_folder,
                                                  transform=transf_pcl, select_steps=select_steps,
                                                  seq_len=seq_len)
    elif mode == 'val':
        fields['points'] = pts_iou_field(p_folder, transform=transf_pt_val,
                                         all_steps=True, seq_len=seq_len, select_steps=select_steps,
                                         unpackbits=unpackbits)
        fields['points_mesh'] = pts_corr_field(pcl_folder,
                                               transform=transf_pcl_val, select_steps=select_steps,
                                               seq_len=seq_len)
    elif mode == 'test' and generate_interpolate:
        fields['mesh'] = data.MeshSubseqField(mesh_folder, seq_len=seq_len,
                                              only_end_points=True)
    return fields
