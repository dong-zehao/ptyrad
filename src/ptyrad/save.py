# Saving

import os

import numpy as np
import torch
from tifffile import imwrite

from ptyrad.utils import get_date, normalize_by_bit_depth, vprint


def make_save_dict(output_path, model, params, optimizer, niter, indices, batch_losses):
    ''' Make a dict to save relevant paramerers '''
    
    avg_losses = {name: np.mean(values) for name, values in batch_losses.items()}
    avg_iter_t = np.mean(model.iter_times)
    
    # While it might seem redundant to save bothe `params` and lots of `model_attributes`,    
    # one should note that `params` only stores the initial value from params files,
    # the actual values used for reconstuction such as N_scan_slow, N_scan_fast, dx, dk, Npix, N_scans could be different from initial value due to the meas_crop, meas_resample
    # the model behavior and learning rates could also be different from the initial params dict if the user
    # run the reconstuction with manually modified `model_params` in the detailed walkthrough notebook
    
    # Postprocess the opt_probe back to complex view
    optimizable_tensors = {}
    for name, tensor in model.optimizable_tensors.items():
        optimizable_tensors[name] = tensor.detach().clone()
        if name == 'probe':
            optimizable_tensors['probe'] = model.get_complex_probe_view().detach().clone()
        
    save_dict = {
                'output_path'           : output_path,
                'optimizable_tensors'   : optimizable_tensors,
                'optim_state_dict'      : optimizer.state_dict() if 'optim_state' in params['recon_params']['save_result'] else None,
                'params'                : params, 
                'model_attributes': # Have to do this explicit saving because I want specific fields but don't want the enitre model with grids and other redundant info
                    {'detector_blur_std': model.detector_blur_std,
                     'obj_preblur_std'  : model.obj_preblur_std,
                     'start_iter'       : model.start_iter,
                     'lr_params'        : model.lr_params,
                     'omode_occu'       : model.omode_occu,
                     'H'                : model.H,
                     'N_scan_slow'      : model.N_scan_slow,
                     'N_scan_fast'      : model.N_scan_fast,
                     'crop_pos'         : model.crop_pos,
                     'slice_thickness'  : model.slice_thickness,
                     'dx'               : model.dx,
                     'dk'               : model.dk,
                     'scan_affine'      : model.scan_affine,
                     'tilt_obj'         : model.tilt_obj,
                     'shift_probes'     : model.shift_probes,
                     'probe_int_sum'    : model.probe_int_sum
                     },
                'loss_iters'            : model.loss_iters,
                'iter_times'            : model.iter_times,
                'dz_iters'              : model.dz_iters,
                'avg_iter_t'            : avg_iter_t,
                'niter'                 : niter,
                'indices'               : indices,
                'batch_losses'          : batch_losses,
                'avg_losses'            : avg_losses
                }
    
    return save_dict

def make_output_folder(output_dir, indices, init_params, recon_params, model, constraint_params, loss_params, recon_dir_affixes=['lr', 'constraint', 'model', 'loss', 'init'], verbose=True):
    ''' 
    Generate the output folder given indices, recon_params, model, constraint_params, and loss_params 
    '''

    output_path  = output_dir
    illumination = init_params['probe_illum_type']
    meas_flipT   = init_params['meas_flipT']
    indices_mode = recon_params['INDICES_MODE'].get('mode')
    group_mode   = recon_params['GROUP_MODE']
    batch_size   = recon_params['BATCH_SIZE'].get('size') * recon_params['BATCH_SIZE'].get('grad_accumulation') # Affix the effective batch size
    prefix_date  = recon_params['prefix_date']
    prefix       = recon_params['prefix']
    postfix      = recon_params['postfix']
    pmode        = model.get_complex_probe_view().size(0)
    dp_size      = model.get_complex_probe_view().size(-1)
    obj_shape    = model.opt_objp.shape
    probe_lr     = format(model.lr_params['probe'],            '.0e').replace("e-0", "e-") if model.lr_params['probe'] !=0 else 0
    objp_lr      = format(model.lr_params['objp'],             '.0e').replace("e-0", "e-") if model.lr_params['objp'] !=0 else 0
    obja_lr      = format(model.lr_params['obja'],             '.0e').replace("e-0", "e-") if model.lr_params['obja'] !=0 else 0
    tilt_lr      = format(model.lr_params['obj_tilts'],        '.0e').replace("e-0", "e-") if model.lr_params['obj_tilts'] !=0 else 0
    dz_lr        = format(model.lr_params['slice_thickness'],  '.0e').replace("e-0", "e-") if model.lr_params['slice_thickness'] !=0 else 0
    pos_lr       = format(model.lr_params['probe_pos_shifts'], '.0e').replace("e-0", "e-") if model.lr_params['probe_pos_shifts'] !=0 else 0
    scan_affine  = model.scan_affine # Note that scan_affine could be None
    init_tilts   = model.opt_obj_tilts.detach().cpu().numpy()
    optimizer_str   = model.optimizer_params['name']
    start_iter_dict = model.start_iter

    # Preprocess prefix and postfix
    prefix  = prefix + '_' if prefix  != '' else ''
    postfix = '_'+ postfix if postfix != '' else ''
    if prefix_date:
        prefix = get_date() + '_' + prefix 
    
    # Setup basic params   
    output_path  = output_dir + "/" + prefix + f"{indices_mode}_N{len(indices)}_dp{dp_size}"
    
    # Attach meas flipping
    if meas_flipT is not None:
        output_path = output_path + '_flipT' + ''.join(str(x) for x in meas_flipT)
    
    # Attach recon mode and pmode 
    output_path += f"_{group_mode}{batch_size}_p{pmode}"
    
    # Attach obj shape and dz
    output_path += f"_{obj_shape[0]}obj_{obj_shape[1]}slice"
    if obj_shape[1] != 1:
        slice_thickness = model.slice_thickness.detach().cpu().numpy().round(2) # This is the initialized slice thickness
        output_path += f"_dz{slice_thickness:.3g}"
    
    # Attach optimizer name (optional)
    if 'optimizer' in recon_dir_affixes:
        output_path += f"_{optimizer_str}"
    
    # Attach start_iter (optional)
    if 'start_iter' in recon_dir_affixes:
        if start_iter_dict['probe'] is not None and start_iter_dict['probe'] > 1:
            output_path += f"_ps{start_iter_dict['probe']}"
        if start_iter_dict['obja'] is not None and start_iter_dict['obja'] > 1:
            output_path += f"_oas{start_iter_dict['obja']}"
        if start_iter_dict['objp'] is not None and start_iter_dict['objp'] > 1:
            output_path += f"_ops{start_iter_dict['objp']}"
        if start_iter_dict['probe_pos_shifts'] is not None and start_iter_dict['probe_pos_shifts'] > 1:
            output_path += f"_ss{start_iter_dict['probe_pos_shifts']}"
        if start_iter_dict['obj_tilts'] is not None and start_iter_dict['obj_tilts'] > 1:
            output_path += f"_ts{start_iter_dict['obj_tilts']}"
        if start_iter_dict['slice_thickness'] is not None and start_iter_dict['slice_thickness'] > 1:
            output_path += f"_dzs{start_iter_dict['slice_thickness']}"
    
    # Attach learning rate (optional)
    if 'lr' in recon_dir_affixes:
        if probe_lr != 0:
            output_path += f"_plr{probe_lr}"
        if obja_lr != 0:
            output_path += f"_oalr{obja_lr}"
        if objp_lr != 0:
            output_path += f"_oplr{objp_lr}"
        if pos_lr != 0:
            output_path += f"_slr{pos_lr}" 
        if tilt_lr != 0:
            output_path += f"_tlr{tilt_lr}"
        if dz_lr != 0:
            output_path += f"_dzlr{dz_lr}"
            
    # Attach model params (optional)
    if 'model' in recon_dir_affixes:    
        if model.obj_preblur_std is not None and model.obj_preblur_std != 0:
            output_path += f"_opreb{model.obj_preblur_std}"
            
        if model.detector_blur_std is not None and model.detector_blur_std != 0:
            output_path += f"_dpblur{model.detector_blur_std}"
    
    # Attach constraint params (optional)
    if 'constraint' in recon_dir_affixes:
        if constraint_params['kr_filter']['freq'] is not None:
            obj_type = constraint_params['kr_filter']['obj_type']
            kr_str = {'both': 'kr', 'amplitude': 'kra', 'phase': 'krp'}.get(obj_type)
            radius = constraint_params['kr_filter']['radius']
            output_path += f"_{kr_str}f{radius}"
        
        if constraint_params['kz_filter']['freq'] is not None:
            obj_type = constraint_params['kz_filter']['obj_type']
            kz_str = {'both': 'kz', 'amplitude': 'kza', 'phase': 'kzp'}.get(obj_type)
            beta = constraint_params['kz_filter']['beta']
            output_path += f"_{kz_str}f{beta}"
            
        if constraint_params['obj_rblur']['freq'] is not None and constraint_params['obj_rblur']['std'] != 0:
            obj_type = constraint_params['obj_rblur']['obj_type']
            obj_str = {'both': 'o', 'amplitude': 'oa', 'phase': 'op'}.get(obj_type)
            output_path += f"_{obj_str}rblur{constraint_params['obj_rblur']['std']}"

        if constraint_params['obj_zblur']['freq'] is not None and constraint_params['obj_zblur']['std'] != 0:
            obj_type = constraint_params['obj_zblur']['obj_type']
            obj_str = {'both': 'o', 'amplitude': 'oa', 'phase': 'op'}.get(obj_type)
            output_path += f"_{obj_str}zblur{constraint_params['obj_zblur']['std']}"
        
        if constraint_params['complex_ratio']['freq'] is not None:
            obj_type = constraint_params['complex_ratio']['obj_type']
            obj_str = {'both': 'o', 'amplitude': 'oa', 'phase': 'op'}.get(obj_type)
            alpha1 = round(constraint_params['complex_ratio']['alpha1'],2)
            alpha2 = round(constraint_params['complex_ratio']['alpha2'],2)
            output_path += f"_{obj_str}cplx{alpha1}_{alpha2}"
        
        if constraint_params['mirrored_amp']['freq'] is not None:
            scale = round(constraint_params['mirrored_amp']['scale'],2)
            power = round(constraint_params['mirrored_amp']['power'],2)
            output_path += f"_mamp{scale}_{power}"
        
        if constraint_params['obja_thresh']['freq'] is not None:
            output_path += f"_oathr{round(constraint_params['obja_thresh']['thresh'][0],2)}"
        
        if constraint_params['objp_postiv']['freq'] is not None:
            mode  = constraint_params['objp_postiv'].get('mode', 'clip_neg')
            mode_str = 's' if mode == 'subtract_min' else 'c'
            relax = constraint_params['objp_postiv']['relax']
            relax_str = '' if relax == 0 else f'{round(relax,2)}'
            output_path += f"_opos{mode_str}{relax_str}"
        
        if constraint_params['tilt_smooth']['freq'] is not None:
            output_path += f"_tsm{round(constraint_params['tilt_smooth']['std'],2)}"
            
        if constraint_params['probe_mask_k']['freq'] is not None:
            output_path += f"_pmk{round(constraint_params['probe_mask_k']['radius'],2)}"

    # Attach loss params (optional)
    if 'loss' in recon_dir_affixes:    
        if loss_params['loss_single']['state']:
            output_path += f"_sng{round(loss_params['loss_single']['weight'],2)}"

        if loss_params['loss_poissn']['state']:
            output_path += f"_psn{round(loss_params['loss_poissn']['weight'],2)}"

        if loss_params['loss_pacbed']['state']:
            output_path += f"_pcb{round(loss_params['loss_pacbed']['weight'],2)}"
        
        if loss_params['loss_sparse']['state']:
            output_path += f"_spr{round(loss_params['loss_sparse']['weight'],2)}"

        if loss_params['loss_simlar']['state']:
            output_path += f"_sml{round(loss_params['loss_simlar']['weight'],2)}"

    # # Attach init params (optional)
    if 'init' in recon_dir_affixes:
        if illumination == 'electron':
            init_conv_angle = init_params['probe_conv_angle']
            init_defocus    = init_params['probe_defocus']
            init_c3    = init_params['probe_c3']
            init_c5    = init_params['probe_c5']
            output_path += f"_ca{init_conv_angle:.3g}"
            output_path += f"_df{init_defocus:.3g}"
            if init_c3 != 0:
                output_path += f"_c3{format(init_c3, '.0e')}"
            if init_c5 != 0:
                output_path += f"_c5{format(init_c5, '.0e')}"
        elif illumination =='xray':
            init_Ls = init_params['Ls']
            output_path += f"_Ls{init_Ls* 1e9:.0f}"
        else:
            raise KeyError(f"init_params['probe_illum_type'] = {illumination} not implemented yet, please use either 'electron' or 'xray'!")
            
    if scan_affine is not None:
        affine_str = '_'.join(f'{x:.2g}' for x in scan_affine)
        output_path += f"_aff{affine_str}"
    
    if np.any(init_tilts):
        tilts_str = '_'.join(f'{x:.2g}' for x in init_tilts.ravel())
        output_path += f"_tilt{tilts_str}"
    
    output_path += postfix
    
    os.makedirs(output_path, exist_ok=True)
    vprint(f"output_path = '{output_path}' is generated!", verbose=verbose)
    return output_path

def copy_params_to_dir(params_path, output_dir, params=None, verbose=True):
    """
    Copies the params file to the output directory if it exists. If the params file does not exist,
    it dumps the provided params dictionary to a YAML file in the output directory.

    Args:
        params_path (str): Path to the params file (can be None if params are programmatically generated).
        output_dir (str): Directory where the params file or YAML dump will be saved.
        params (dict, optional): The programmatically generated params dictionary to save if no file exists.
        verbose (bool): Whether to print verbose messages.
    """
    import shutil
    import yaml
    import os

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if params_path and os.path.isfile(params_path):
        # If the params file exists, copy it to the output directory
        file_name = os.path.basename(params_path)
        output_path = os.path.join(output_dir, file_name)
        shutil.copy2(params_path, output_path)
        vprint(" ")
        vprint(f"### Successfully copy '{file_name}' to '{output_dir}' ###", verbose=verbose)

    elif params is not None:
        # If no file exists, dump the params dictionary to a YAML file
        output_path = os.path.join(output_dir, "params_dumped.yml")
        with open(output_path, "w") as f:
            yaml.safe_dump(params, f, sort_keys=False)
        vprint(" ")
        vprint(f"### No params file found. Dumped params dictionary to '{output_path}' ###")

    else:
        # If neither a file nor params are provided, skip with a warning
        vprint(" ")
        vprint("### Warning: No params file found and no params dictionary provided. Skipping. ###", verbose=verbose)

@torch.compiler.disable
def save_results(output_path, model, params, optimizer, niter, indices, batch_losses, collate_str=''):
    
    save_result_list = params['recon_params'].get('save_result', ['model', 'obj', 'probe'])
    result_modes = params['recon_params'].get('result_modes')
    iter_str = '_iter' + str(niter).zfill(4)
    
    if 'model' in save_result_list:
        save_dict = make_save_dict(output_path, model, params, optimizer, niter, indices, batch_losses)
        torch.save(save_dict, os.path.join(output_path, f"model{collate_str}{iter_str}.pt"))
    probe      = model.get_complex_probe_view() 
    probe_amp  = probe.reshape(-1, probe.size(-1)).t().abs().detach().cpu().numpy()
    probe_prop = model.get_propagated_probe([0]).permute(0,2,1,3)
    shape      = probe_prop.shape
    prop_p_amp = probe_prop.reshape(shape[0]*shape[1], shape[2]*shape[3]).abs().detach().cpu().numpy()
    objp       = model.opt_objp.detach().cpu().numpy()
    obja       = model.opt_obja.detach().cpu().numpy()
    # omode_occu = model.omode_occu # Currently not used but we'll need it when omode_occu != 'uniform'
    omode      = model.opt_objp.size(0)
    zslice     = model.opt_objp.size(1)
    crop_pos   = model.crop_pos[indices].detach().cpu().numpy() + np.array(probe.shape[-2:])//2
    y_min, y_max = crop_pos[:,0].min(), crop_pos[:,0].max()
    x_min, x_max = crop_pos[:,1].min(), crop_pos[:,1].max()
    
    for bit in result_modes['bit']:
        if bit == '8':
            bit_str = '_08bit'
        elif bit == '16':
            bit_str = '_16bit'
        elif bit == '32':
            bit_str = '_32bit'
        elif bit == 'raw':
            bit_str = ''
        else:
            bit_str = ''
        if 'probe' in save_result_list:
            imwrite(os.path.join(output_path, f"probe_amp{bit_str}{collate_str}{iter_str}.tif"), normalize_by_bit_depth(probe_amp, bit))
        if 'probe_prop' in save_result_list:
            imwrite(os.path.join(output_path, f"probe_prop_amp{bit_str}{collate_str}{iter_str}.tif"), normalize_by_bit_depth(prop_p_amp, bit))
        for fov in result_modes['FOV']:
            if fov == 'crop':
                fov_str = '_crop'
                objp_crop = objp[:, :, y_min-1:y_max, x_min-1:x_max]
                obja_crop = obja[:, :, y_min-1:y_max, x_min-1:x_max]
            elif fov == 'full':
                fov_str = ''
                objp_crop = objp
                obja_crop = obja
            else:
                fov_str = ''
                objp_crop = objp
                obja_crop = obja
                
            postfix_str = fov_str + bit_str + collate_str + iter_str
                
            if any(keyword in save_result_list for keyword in ['obj', 'objp', 'object']):
                # TODO: For omode_occu != 'uniform', we should do a weighted sum across omode instead
                
                for dim in result_modes['obj_dim']:
                    
                    if omode == 1 and zslice == 1:
                        if dim == 2: 
                            imwrite(os.path.join(output_path, f"objp{postfix_str}.tif"),              normalize_by_bit_depth(objp_crop[0,0], bit))
                    elif omode == 1 and zslice > 1:
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"objp_zstack{postfix_str}.tif"),       normalize_by_bit_depth(objp_crop[0,:], bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"objp_zsum{postfix_str}.tif"),         normalize_by_bit_depth(objp_crop[0,:].sum(0), bit))
                    elif omode > 1 and zslice == 1:
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"objp_ostack{postfix_str}.tif"),       normalize_by_bit_depth(objp_crop[:,0], bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"objp_omean{postfix_str}.tif"),        normalize_by_bit_depth(objp_crop[:,0].mean(0), bit))
                            imwrite(os.path.join(output_path, f"objp_ostd{postfix_str}.tif"),         normalize_by_bit_depth(objp_crop[:,0].std(0), bit))
                    else:
                        if dim == 4:
                            imwrite(os.path.join(output_path, f"objp_4D{postfix_str}.tif"),           normalize_by_bit_depth(objp_crop[:,:], bit))
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"objp_ostack_zsum{postfix_str}.tif"),  normalize_by_bit_depth(objp_crop[:,:].sum(1), bit))
                            imwrite(os.path.join(output_path, f"objp_omean_zstack{postfix_str}.tif"), normalize_by_bit_depth(objp_crop[:,:].mean(0), bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"objp_omean_zsum{postfix_str}.tif"),   normalize_by_bit_depth(objp_crop[:,:].mean(0).sum(0), bit))
                            
            if any(keyword in save_result_list for keyword in ['obja']):
                # TODO: For omode_occu != 'uniform', we should do a weighted sum across omode instead
                
                for dim in result_modes['obj_dim']:
                    
                    if omode == 1 and zslice == 1:
                        if dim == 2: 
                            imwrite(os.path.join(output_path, f"obja{postfix_str}.tif"),              normalize_by_bit_depth(obja_crop[0,0], bit))
                    elif omode == 1 and zslice > 1:
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"obja_zstack{postfix_str}.tif"),       normalize_by_bit_depth(obja_crop[0,:], bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"obja_zmean{postfix_str}.tif"),         normalize_by_bit_depth(obja_crop[0,:].mean(0), bit))
                            imwrite(os.path.join(output_path, f"obja_zprod{postfix_str}.tif"),         normalize_by_bit_depth(obja_crop[0,:].prod(0), bit))
                    elif omode > 1 and zslice == 1:
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"obja_ostack{postfix_str}.tif"),       normalize_by_bit_depth(obja_crop[:,0], bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"obja_omean{postfix_str}.tif"),        normalize_by_bit_depth(obja_crop[:,0].mean(0), bit))
                            imwrite(os.path.join(output_path, f"obja_ostd{postfix_str}.tif"),         normalize_by_bit_depth(obja_crop[:,0].std(0), bit))
                    else:
                        if dim == 4:
                            imwrite(os.path.join(output_path, f"obja_4D{postfix_str}.tif"),           normalize_by_bit_depth(obja_crop[:,:], bit))
                        if dim == 3:
                            imwrite(os.path.join(output_path, f"obja_ostack_zmean{postfix_str}.tif"),  normalize_by_bit_depth(obja_crop[:,:].mean(1), bit))
                            imwrite(os.path.join(output_path, f"obja_ostack_zprod{postfix_str}.tif"),  normalize_by_bit_depth(obja_crop[:,:].prod(1), bit))
                            imwrite(os.path.join(output_path, f"obja_omean_zstack{postfix_str}.tif"), normalize_by_bit_depth(obja_crop[:,:].mean(0), bit))
                        if dim == 2:
                            imwrite(os.path.join(output_path, f"obja_omean_zmean{postfix_str}.tif"),   normalize_by_bit_depth(obja_crop[:,:].mean(0).mean(0), bit))
                            imwrite(os.path.join(output_path, f"obja_omean_zprod{postfix_str}.tif"),   normalize_by_bit_depth(obja_crop[:,:].mean(0).prod(0), bit))