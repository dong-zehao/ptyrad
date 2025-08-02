## Defining PtychoAD class for the optimization object
from math import prod
import torch
from torch.fft import fft2, ifft2, fftfreq
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur

from ptyrad.forward import multislice_forward_model_vec_all
from ptyrad.utils import imshift_batch, torch_phasor, vprint

# The obj_ROI_grid is modified from precalculation to on-the-fly generation for memory consumption
# It has very little performance impact but saves lots of memory for large 4D-STEM data
# Added create_grids and print_model_summary for readability and decoupling
# set_optimizer function is called at the end of the initializaiton, while this can also be called if you want to update the optimizer params without initializing the object
# obj optimization is now split into objp and obja
# mixed object modes are normalized by the init_omode_occu. By design this is a fixed value because optimizing omode_occu with obj simultaneously could be unstable
# obj_ROI cropping is done with vectorization and the obj_ROI_grid is only generated once
# probe with sub-px shifts are calculated only when probe_pos_shifts are enables for optimization
# All the sub-px shifted probes in a batch are processed together with vectorizaiton
# Likewise, the multislice forward model is also fully vectorized across samples (in batch), pmode, and omode
# Note that it's possible to reduce the peak-memory consumption by reducing the level of vectorizaiton and roll back to for loops
# Lastly, the forward pass of this model would output the dp_fwd (N, Ky, Kx) and objp_patches (N, omode, Nz, Ny, Nx) in float32 for later loss calculation

class PtychoAD(torch.nn.Module):
    """
    Main optimization class for ptychographic reconstruction using automatic differentiation (AD).

    This class is responsible for initializing the model parameters, setting up the optimizer, and
    performing forward passes to compute diffraction patterns based on the given input indices.

    Attributes:
        device (str): Device to run computations on ('cuda:0' by default).
        verbose (bool): If True, prints model summary (True by default).
        detector_blur_std (float): Standard deviation for detector blur, or None if no blur.
        obj_preblur_std (float): Standard deviation for object pre-blur, or None if no pre-blur.
        lr_params (dict): Learning rate parameters for optimizable tensors.
        wd_params (dict): Weight decay parameters for optimizable tensors.
        opt_obja (torch.Tensor): Amplitude of the object.
        opt_objp (torch.Tensor): Phase of the object.
        opt_obj_tilts (torch.Tensor): Tilts of the object.
        opt_probe (torch.Tensor): Probe function.
        opt_probe_pos_shifts (torch.Tensor): Shifts for the probe positions.
        omode_occu (torch.Tensor): Occupation mode.
        H (torch.Tensor): Propagator matrix.
        measurements (torch.Tensor): Measurements for the ptychographic reconstruction.
        N_scan_slow (torch.Tensor): Number of scans in the slow direction.
        N_scan_fast (torch.Tensor): Number of scans in the fast direction.
        crop_pos (torch.Tensor): Cropping positions.
        slice_thickness (torch.Tensor): slice thickness (dz) parameter.
        dx (torch.Tensor): Pixel size in the x direction.
        dk (torch.Tensor): K-space sampling interval.
        scan_affine (affine.Affine): Affine transformation for scan.
        tilt_obj (bool): Whether object tilts are being optimized.
        shift_probes (bool): Whether probe shifts are being optimized.
        probe_int_sum (torch.Tensor): Sum of squared probe intensities.
        optimizable_tensors (dict): Dictionary of tensors that can be optimized.

    Args:
        init_variables (dict): Dictionary of initial variables required for the model.
        model_params (dict): Dictionary of model parameters including learning rates and blur stds.
        device (str): Device to run computations on. Default is 'cuda:0'.
        verbose (bool): If True, prints model summary. Default is True.

    Methods:
        create_grids():
            Creates the grids for object ROI and probe shifts.
        set_optimizer(lr_params, verbose=True):
            Sets up the optimizer with learning rate parameters.
        print_model_summary():
            Prints a summary of the model's optimizable variables and statistics.
        get_obj_ROI(indices):
            Retrieves object ROI based on integer coordinates.
        get_probes(indices):
            Retrieves probe functions for each position.
        get_propagators(indices):
            Retrieves propagator matrices for each position.
        get_measurements(indices=None):
            Retrieves measurements based on input indices.
        forward(indices):
            Performs the forward pass and computes diffraction patterns for given indices.
    """

    def __init__(self, init_variables, model_params, device='cuda', verbose=True):
        super(PtychoAD, self).__init__()
        with torch.no_grad():
            
            vprint('### Initializing PtychoAD model ###', verbose=verbose)
            
            # Setup model behaviors
            self.device                 = device
            self.verbose                = verbose
            self.detector_blur_std      = model_params['detector_blur_std']
            self.obj_preblur_std        = model_params['obj_preblur_std']
            if init_variables.get('on_the_fly_meas_padded', None) is not None:
                self.meas_padded        = torch.tensor(init_variables['on_the_fly_meas_padded'], dtype=torch.float32, device=device)
                self.meas_padded_idx    = torch.tensor(init_variables['on_the_fly_meas_padded_idx'], dtype=torch.int32, device=device)
            else:
                self.meas_padded        = None
            self.meas_scale_factors     = init_variables.get('on_the_fly_meas_scale_factors', None)

            # Parse the learning rate, start iter and weight decay for optimizable tensors
            start_iter_dict = {}
            lr_dict = {}
            wd_dict = {}
            for key, params in model_params['update_params'].items():
                start_iter_dict[key] = params['start_iter']
                lr_dict[key] = params['lr']
                wd_dict[key] = params.get('weight_decay', 0.0)
            self.optimizer_params       = model_params['optimizer_params']
            self.start_iter             = start_iter_dict
            self.lr_params              = lr_dict
            self.wd_params              = wd_dict

            # Optimizable parameters
            self.opt_obja               = nn.Parameter(torch.abs(torch.tensor(init_variables['obj'],    device=device)).to(torch.float32))
            self.opt_objp               = nn.Parameter(torch.angle(torch.tensor(init_variables['obj'],  device=device)).to(torch.float32))
            self.opt_obj_tilts          = nn.Parameter(torch.tensor(init_variables['obj_tilts'],                dtype=torch.float32, device=device))
            self.opt_slice_thickness    = nn.Parameter(torch.tensor(init_variables['slice_thickness'],          dtype=torch.float32, device=device))
            self.opt_probe              = nn.Parameter(torch.view_as_real(torch.tensor(init_variables['probe'], dtype=torch.complex64, device=device))) # The `torch.view_as_real` allows correct handling of DDP via NCCL even in PyTorch 2.4
            self.opt_probe_pos_shifts   = nn.Parameter(torch.tensor(init_variables['probe_pos_shifts'],         dtype=torch.float32, device=device))
            
            # Buffers are used during forward pass
            self.register_buffer      ('omode_occu',      torch.tensor(init_variables['omode_occu'],       dtype=torch.float32, device=device))
            self.register_buffer      ('H',               torch.tensor(init_variables['H'],                dtype=torch.complex64, device=device))
            self.register_buffer      ('measurements',    torch.tensor(init_variables['measurements'],     dtype=torch.float32, device=device))
            self.register_buffer      ('N_scan_slow',     torch.tensor(init_variables['N_scan_slow'],      dtype=torch.int32, device=device))# Saving this for reference, the cropping is based on self.obj_ROI_grid.
            self.register_buffer      ('N_scan_fast',     torch.tensor(init_variables['N_scan_fast'],      dtype=torch.int32, device=device))# Saving this for reference, the cropping is based on self.obj_ROI_grid.
            self.register_buffer      ('crop_pos',        torch.tensor(init_variables['crop_pos'],         dtype=torch.int32, device=device))# Saving this for reference, the cropping is based on self.obj_ROI_grid.
            self.register_buffer      ('slice_thickness', torch.tensor(init_variables['slice_thickness'],  dtype=torch.float32, device=device))# Saving this for reference
            self.register_buffer      ('dx',              torch.tensor(init_variables['dx'],               dtype=torch.float32, device=device))# Saving this for reference
            self.register_buffer      ('dk',              torch.tensor(init_variables['dk'],               dtype=torch.float32, device=device))# Saving this for reference
            self.register_buffer      ('lambd',           torch.tensor(init_variables['lambd'],            dtype=torch.float32, device=device))
            
            self.scan_affine            = init_variables['scan_affine']                                                           # Saving this for reference
            self.tilt_obj               = bool(self.lr_params['obj_tilts']        != 0 or torch.any(self.opt_obj_tilts))          # Set tilt_obj to True if lr_params['obj_tilts'] is not 0 or we have any none-zero tilt values
            self.shift_probes           = bool(self.lr_params['probe_pos_shifts'] != 0)                                           # Set shift_probes to True if lr_params['probe_pos_shifts'] is not 0
            self.change_thickness       = bool(self.lr_params['slice_thickness']  != 0)
            self.probe_int_sum          = self.get_complex_probe_view().abs().pow(2).sum() # This is only used for the `fix_probe_int`
            self.loss_iters             = []
            self.iter_times             = []
            self.dz_iters               = []
            self.avg_tilt_iters         = []

            # Create grids for shifting
            self.create_grids()

            # Create a dictionary to store the optimizable tensors
            self.optimizable_tensors = {
                'obja'            : self.opt_obja,
                'objp'            : self.opt_objp,
                'obj_tilts'       : self.opt_obj_tilts,
                'slice_thickness' : self.opt_slice_thickness,
                'probe'           : self.opt_probe,
                'probe_pos_shifts': self.opt_probe_pos_shifts}
            self.create_optimizable_params_dict(self.lr_params, self.wd_params, self.verbose)

            vprint('### Done initializing PtychoAD model ###', verbose=verbose)
            vprint(' ', verbose=verbose)
            
    def get_complex_probe_view(self):
        """ Retrieve complex view of the probe """
        # This is a post-processing to ensure minimal code changes in PtyRAD for the DDP (multiGPU) via NCCL due to limited support for Complex value
        return torch.view_as_complex(self.opt_probe)
        
    def create_grids(self):
        """ Create the grid for obj_ROI and shift_probes in a vectorized approach """
        # Note that the Noy, Nox, roy, rox, shift_object_grid are pre-generated for potential future usage of sub-px object shifts
        # Currently (2024.04.24) only the shift_probes_grid, rpy_grid, rpx_grid are used for sub-px shifts and obj_ROI selection
        # 2024.11.18 added propagator_grid and propagator_tilt_grid for Fresnel propagator (optimizable slice thickness and tilts)
        
        device = self.device
        probe = self.get_complex_probe_view()
        Npy, Npx = probe.shape[-2:] # Number of probe pixels in y and x directions
        Noy, Nox = self.opt_objp.shape[-2:] # Number of object pixels in y and x directions, 
        
        # Grids for Fresnel propagator
        ygrid = (torch.arange(-Npy // 2, Npy // 2, device=device) + 0.5) / Npy
        xgrid = (torch.arange(-Npx // 2, Npx // 2, device=device) + 0.5) / Npx
        ky = 2 * torch.pi * ygrid / self.dx
        kx = 2 * torch.pi * xgrid / self.dx
        Ky, Kx = torch.meshgrid(ky, kx, indexing="ij")
        self.propagator_grid = torch.stack([Ky,Kx], dim=0) # (2,Ky,Kx), k-space grid for Fresnel propagator
        
        # Grids for propagator tilt (the units are different from the above grid)
        ky = fftfreq(Npy, 1 / Npy, device=device) * self.dk # dk in 1/Ang
        kx = fftfreq(Npx, 1 / Npx, device=device) * self.dk
        Ky, Kx = torch.meshgrid(ky, kx, indexing='ij')
        self.propagator_tilt_grid = torch.stack([Ky,Kx], dim=0) # (2,Ky,Kx), k-space grid for Fresnel propagator tilt

        # Grids for shifting porbes and obj_ROI selection
        rpy, rpx = torch.meshgrid(torch.arange(Npy, dtype=torch.int32, device=device), 
                                  torch.arange(Npx, dtype=torch.int32, device=device), indexing='ij') # real space grid for probe in y and x directions
        roy, rox = torch.meshgrid(torch.arange(Noy, dtype=torch.int32, device=device), 
                                  torch.arange(Nox, dtype=torch.int32, device=device), indexing='ij') # real space grid for object in y and x directions
        
        self.shift_probes_grid = torch.stack([rpy/Npy, rpx/Npx], dim=0) # (2,Npy,Npx), normalized k-space grid stack for sub-px probe shifting 
        self.shift_object_grid = torch.stack([roy/Noy, rox/Nox], dim=0) # (2,Noy,Nox), normalized k-space grid stack for sub-px object shifting (Implemented for completeness, not used in PtyRAD)
        
        self.rpy_grid = rpy # real space grid with y-indices spans across probe extent
        self.rpx_grid = rpx
        self.roy_grid = roy # real space grid with y-indices spans across object extent
        self.rox_grid = rox

    def create_optimizable_params_dict(self, lr_params, wd_params, verbose=True):
        """ Sets the optimizer with lr_params and wd_params"""
        # # Use this to edit learning rate if needed some refinement

        # model.set_optimizer(lr_params={'obja'            : 5e-4,
        #                                'objp'            : 5e-4,
        #                                'obj_tilts'       : 1e-4,
        #                                'probe'           : 1e-4, 
        #                                'probe_pos_shifts': 1e-4})
        # optimizer=torch.optim.Adam(model.optimizer_params)
        
        self.lr_params = lr_params
        self.wd_params = wd_params
        self.optimizable_params = []
        for param_name, lr in lr_params.items():
            if param_name not in self.optimizable_tensors:
                raise KeyError(f"Warning: '{param_name}' is not a valid parameter name, check your lr_params.")
            else:
                self.optimizable_tensors[param_name].requires_grad = (lr != 0) # Set requires_grad based on learning rate
                if lr != 0:
                    self.optimizable_params.append({'params': [self.optimizable_tensors[param_name]], 'lr': lr, 'weight_decay': wd_params.get(param_name, 0.0)})
        if verbose:
            self.print_model_summary()
        
    def print_model_summary(self):
        # Set all the print as vprint so that it'll only print once in DDP, the actual `if verbose` is set outside of the function
        vprint('### PtychoAD optimizable variables ###')
        for name, tensor in self.optimizable_tensors.items():
            vprint(f"{name.ljust(16)}: {str(tensor.shape).ljust(32)}, {str(tensor.dtype).ljust(16)}, device:{tensor.device}, grad:{str(tensor.requires_grad).ljust(5)}, lr:{self.lr_params[name]:.0e}, weight_decay:{self.wd_params.get(name, 0.0):.0e}")
        total_var = sum(tensor.numel() for _, tensor in self.optimizable_tensors.items() if tensor.requires_grad)
        # When you create a new model, make sure to pass the optimizer_params to optimizer using "optimizer = torch.optim.Adam(model.optimizer_params)"
        vprint(" ")        
        
        vprint('### Optimizable variables statitsics ###')
        vprint(f"Total measurement values  : {self.measurements.numel():,d}")
        vprint(f"Total optimizing variables: {total_var:,d}")
        vprint(f"Overdetermined ratio      : {self.measurements.numel()/total_var:.2f}")
        vprint(" ")
        
        vprint('### Model behavior ###')
        vprint(f"Obj preblur               : {True if self.obj_preblur_std is not None else False}")
        vprint(f"Tilt propagator           : {self.tilt_obj}")
        vprint(f"Change slice thickness    : {self.change_thickness}")
        vprint(f"Sub-px probe shift        : {self.shift_probes}")
        vprint(f"Detector blur             : {True if self.detector_blur_std is not None else False}")
        vprint(f"On-the-fly meas padding   : {True if self.meas_padded is not None else False}")
        vprint(f"On-the-fly meas resample  : {True if self.meas_scale_factors is not None else False}")
        vprint(" ")
    
    def get_obj_ROI(self, indices):
        """ Get object ROI with integer coordinates """
        # It's strongly recommended to do integer version of get_obj_ROI
        # opt_obj.shape = (B,D,H,W,C) = (omode,D,H,W,2)
        # object_patches = (N,B,D,H,W,2), N is the additional sample index within the input batch, B is now used for omode.
        
        # rpy_grid is the y-grid (Ny,Nx), by adding the y coordinates from init_crop_pos (N,1) in a broadcast way, it becomes (N,Ny,Nx)
        # obj_ROI_grid_y = (N,Ny,Nx)
        
        opt_obj = torch.stack([self.opt_obja, self.opt_objp], dim=-1)
        obj_ROI_grid_y = self.rpy_grid[None,:,:] + self.crop_pos[indices, None, None, 0]
        obj_ROI_grid_x = self.rpx_grid[None,:,:] + self.crop_pos[indices, None, None, 1]
        
        object_patches = opt_obj[:,:,obj_ROI_grid_y,obj_ROI_grid_x,:].permute(2,0,1,3,4,5)
        return object_patches
    
    def get_probes(self, indices):
        """ Get probes for each position """
        # If you're not trying to optimize probe positions, there's not much point using sub-px shifted stationary probes
        # This function will return a single probe when self.shift_probes = False,
        # and would only be returning multiple sub-px shifted probes if you're optimizing self.opt_probe_pos_shifts

        probe = self.get_complex_probe_view()
        
        if self.shift_probes:
            probes = imshift_batch(probe, shifts = self.opt_probe_pos_shifts[indices], grid = self.shift_probes_grid)
        else:
            probes = torch.broadcast_to(probe, (len(indices), *probe.shape)) # Broadcast a batch dimension, essentially using same probe for all samples
        return probes
    
    def get_propagators(self, indices):
        """ Get propagators for each position """
        # self.tilt_obj is True as long as we're optimizing the opt_obj_tilts or we have non-zero initial tilt values
        # This function will return a single propagator (H) if self.opt_obj_tilts has shape = (1,2) (single tilt_y, tilt_x) 
        # If self.opt_obj_tilts has shape = (N,2), it'll return multiple propagtors stacked at axis 0 (N,Y,X)
        # Note that 0 tilts is numerically equivalent to the H and can be verified by "torch.allclose(model.H, model.get_propagators([0]))"
        
        # If you want to expand the opt_obj_tilts from (1,2) to (N,2), use the following lines
        # model.opt_obj_tilts.data = torch.broadcast_to(model.opt_obj_tilts, [<N_scans>,2]).clone() # Replace <N_scans> with your actual N_scans
        # model.set_optimizer(lr_params = {
        # 'obja'            : 5e-4,
        # 'objp'            : 5e-4,
        # 'obj_tilts'       : 1e-4, 
        # 'probe'           : 1e-4, 
        # 'probe_pos_shifts': 1e-4})
        # optimizer=torch.optim.Adam(model.optimizer_params)
        
        dz  = self.opt_slice_thickness
        
        if self.tilt_obj is True:

            # Setup obj_tilts and the tilt kernel
            if self.opt_obj_tilts.shape[0] == 1: # If there's only 1 global tilt obp_obj_tilts.shape = (1,2)
                obj_tilts = self.opt_obj_tilts
            else:
                obj_tilts = self.opt_obj_tilts[indices]
            tilts_y  = obj_tilts[:,0,None,None] / 1e3 #mrad, tilts_y = (N,Y,X)
            tilts_x  = obj_tilts[:,1,None,None] / 1e3
            Kyt, Kxt = self.propagator_tilt_grid
            
            # tilt is either non-zero or optimizing, thickness is optimizing as well. Have to calculate propagators for each batch
            if self.change_thickness is True:
                k           = 2 * torch.pi / self.lambd
                Ky, Kx      = self.propagator_grid
                H_opt_dz    = torch.fft.ifftshift(torch_phasor(dz * torch.sqrt(k ** 2 - Kx ** 2 - Ky ** 2))) # H has zero frequency at the corner in k-space
                phase_shift = 2 * torch.pi * dz * (Kyt * torch.tan(tilts_y) + Kxt * torch.tan(tilts_x)) # phase_shift is (N,Ky,Kx)
                return H_opt_dz * torch_phasor(phase_shift)

            else: # self.change_thickness is False

                # Thickness is fixed, but obj_tilts is optimizing throughout the iterations
                if self.lr_params['obj_tilts'] != 0:
                    phase_shift = 2 * torch.pi * dz * (Kyt * torch.tan(tilts_y) + Kxt * torch.tan(tilts_x))
                    return self.H * torch_phasor(phase_shift)

                # Thickness is fixed, but obj_tilts is a fixed non-zero value throughout the iterations, so it should only be calculated once
                else:
                    try:
                        return self.H_fixed_tilts
                    except AttributeError:
                        phase_shift = 2 * torch.pi * dz * (Kyt * torch.tan(tilts_y) + Kxt * torch.tan(tilts_x))
                        self.H_fixed_tilts = self.H * torch_phasor(phase_shift)
                    return self.H_fixed_tilts
        
        else: # self.tilt is False

            if self.change_thickness is True:
                # Thickness can be optimized, but tilt is fixed at 0
                k        = 2 * torch.pi / self.lambd
                Ky, Kx   = self.propagator_grid
                H_opt_dz = torch.fft.ifftshift(torch_phasor(dz * torch.sqrt(k ** 2 - Kx ** 2 - Ky ** 2))) # H has zero frequency at the corner in k-space
                return H_opt_dz[None,]
            
            else: # self.change_thickness is False and self.tilt_obj is False:
                return self.H[None,]

    def get_propagated_probe(self, index):
        probe = self.get_probes(index)[0].detach() # (pmode, Ny, Nx)
        H = self.get_propagators(index).detach() # (1, Ny, Nx)
        n_slices = self.opt_objp.shape[1]
        probe_prop = torch.zeros((n_slices, *probe.shape), dtype=probe.dtype, device=probe.device)
        
        psi = probe # (z, pmode, Ny, Nx)
        for n in range(n_slices):
            probe_prop[n] = psi
            psi = ifft2(H[None,] * fft2(psi))
        
        return probe_prop
    
    def get_measurements(self, indices=None):
        """ Get measurements for each position """
        # Return the selected measurements based on input indices
        # If no indices are passed, return the entire measurements ignoring any "on-the-fly" padding/resampling
        
        measurements = self.measurements
        device       = self.device
        dtype        = measurements.dtype
        if self.meas_padded is not None:
            meas_padded  = self.meas_padded
            meas_padded_idx = self.meas_padded_idx
            pad_h1, pad_h2, pad_w1, pad_w2 = meas_padded_idx
        scale_factor = tuple(self.meas_scale_factors) if self.meas_scale_factors is not None else None
        
        if indices is not None:
            measurements = self.measurements[indices]
            
            if self.meas_padded is not None:
                canvas = torch.zeros((measurements.shape[0], *meas_padded.shape[-2:]), dtype=dtype, device=device)
                canvas += meas_padded
                canvas[..., pad_h1:pad_h2, pad_w1:pad_w2] = measurements # Replace the center part with the original meas
                measurements = canvas
            
            if self.meas_scale_factors is not None and any(factor != 1 for factor in scale_factor):
                measurements = torch.nn.functional.interpolate(measurements[None,], scale_factor=scale_factor, mode='bilinear')[0] # 2D interpolate requires 4D input (N, C, H, W)
                measurements = measurements / prod(scale_factor) # This ensures the intensity scale and the integrated intensity are unchanged
            
        else: # Skip the "on-the-fly" operations so it won't throw any CUDA out-of-memory error. All typical PtyRAD usuage would pass get_measurements(batch) so this should be ok.
            if self.meas_padded is not None or self.meas_scale_factors is not None:
                vprint(f"WARNING: 'on-the-fly' measurements padding/resampling detected, but they are ignored because it may cause 'CUDA out-of-memory' when 'get_measurements()' is called without any indices. The original measurement with shape = {self.measurements.shape} is returned instead.")
            measurements = self.measurements
        
        return measurements
        
    def forward(self, indices):
        """ Doing the forward pass and get an output diffraction pattern for each input index """
        # The indices are passed in as an array and representing the whole batch
        # Note that detector blur is a physical process and should be included in the forward method
        # It's a design choice to put it here, instead of putting it under optimization.py
        
        object_patches = self.get_obj_ROI(indices)
        
        if self.obj_preblur_std is not None and self.obj_preblur_std != 0:
            # Permute and reshape approach, this is much faster than the stack/list comprehension version
            obj = object_patches.permute(5,0,1,2,3,4) # Move the r/i to the front
            obj_shape = obj.shape
            obj = obj.reshape(-1, obj_shape[-2], obj_shape[-1])
            object_patches = gaussian_blur(obj, kernel_size=5, sigma=self.obj_preblur_std).reshape(obj_shape).permute(1,2,3,4,5,0)
        
        probes         = self.get_probes(indices)
        propagators    = self.get_propagators(indices)
        dp_fwd         = multislice_forward_model_vec_all(object_patches, self.omode_occu, probes, propagators)
        
        if self.detector_blur_std is not None and self.detector_blur_std != 0:
            dp_fwd = gaussian_blur(dp_fwd, kernel_size=5, sigma=self.detector_blur_std)
        return dp_fwd, object_patches