import io
import logging
import os
from time import perf_counter

import torch
import torch.distributed as dist

def torch_phasor(phase):
    """
    Creates a complex tensor with unit magnitude using the phase.

    Args:
        phase (torch.Tensor): phase angle for the exp(i*theta)
        
    Note:
        This util function is created so torch.compile can properly handle complex tensors,
        because torch.exp(1j*phase) involves the 1j which is actually a Python built-in that can't be traced.
    """
    return torch.polar(torch.ones_like(phase), phase)

# Only used in run_ptyrad.py, might have a better place
def set_accelerator():

    try:
        from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
        dataloader_config  = DataLoaderConfiguration(split_batches=True) # This supress the warning when we do `Accelerator(split_batches=True)`
        kwargs_handlers    = [DistributedDataParallelKwargs(find_unused_parameters=False)] # This avoids the error `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss.` We don't necessarily need this if we carefully register parameters (used in forward) and buffer in the `model`.
        accelerator        = Accelerator(dataloader_config=dataloader_config, kwargs_handlers=kwargs_handlers)
        vprint("### Initializing HuggingFace accelerator ###")
        vprint(f"Accelerator.distributed_type = {accelerator.distributed_type}")
        vprint(f"Accelerator.num_process      = {accelerator.num_processes}")
        vprint(f"Accelerator.mixed_precision  = {accelerator.mixed_precision}")
        
    except ImportError:
        vprint("### HuggingFace accelerator is not available, no multi-GPU or mixed-precision ###")
        accelerator = None
        
    vprint(" ")
    return accelerator

# System level utils
class CustomLogger:
    def __init__(self, log_file='output.log', log_dir='auto', prefix_date=True, prefix_jobid=0, append_to_file=True, show_timestamp=True):
        self.logger = logging.getLogger('PtyRAD')
        self.logger.setLevel(logging.INFO)
        
        # Clear all existing handlers to re-instantiate the logger
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.log_file       = log_file
        self.log_dir        = log_dir
        self.flush_file     = log_file is not None
        self.prefix_date    = prefix_date
        self.prefix_jobid   = prefix_jobid
        self.append_to_file = append_to_file
        self.show_timestamp = show_timestamp

        # Create console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s' if show_timestamp else '%(message)s')
        self.console_handler.setFormatter(formatter)
        
        # Create a buffer for file logs
        self.log_buffer = io.StringIO()
        self.buffer_handler = logging.StreamHandler(self.log_buffer)
        self.buffer_handler.setLevel(logging.INFO)
        self.buffer_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.buffer_handler)
        
        # Print logger information
        vprint("### PtyRAD Logger configuration ###")
        vprint(f"log_file       = '{self.log_file}'. If log_file = None, no log file will be created.")
        vprint(f"log_dir        = '{self.log_dir}'. If log_dir = 'auto', then log will be saved to `output_path` or 'logs/'.")
        vprint(f"flush_file     = {self.flush_file}. Automatically set to True if `log_file is not None`")
        vprint(f"prefix_date    = {self.prefix_date}. If true, a datetime str is prefixed to the `log_file`.")
        vprint(f"prefix_jobid   = '{self.prefix_jobid}'. If not 0, it'll be prefixed to the log file. This is used for hypertune mode with multiple GPUs.")
        vprint(f"append_to_file = {self.append_to_file}. If true, logs will be appended to the existing file. If false, the log file will be overwritten.")
        vprint(f"show_timestamp = {self.show_timestamp}. If true, the printed information will contain a timestamp.")
        vprint(' ')

    def flush_to_file(self, log_dir=None, append_to_file=None):
        """
        Flushes buffered logs to a file based on user-defined file mode (append or write)
        """
        
        # Set log_dir
        if log_dir is None:
            if self.log_dir == 'auto':
                log_dir = 'logs'
            else:
                log_dir = self.log_dir

        # Set file_mode
        if append_to_file is None:
            append_to_file = self.append_to_file
        file_mode = 'a' if append_to_file else 'w'
        
        # Set file name
        log_file = self.log_file
        if self.prefix_jobid != 0:
            log_file = str(self.prefix_jobid).zfill(2) + '_' + log_file
        if self.prefix_date:
            log_file = get_date() + '_' + log_file
        
        show_timestamp = self.show_timestamp
        
        if self.flush_file:
            # Ensure the log directory exists
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, log_file)

            # Write the buffered logs to the specified file
            with open(log_file_path, file_mode) as f:
                f.write(self.log_buffer.getvalue())

            # Clear the buffer
            self.log_buffer.truncate(0)
            self.log_buffer.seek(0)

            # Set up a file handler for future logging to the file
            self.file_handler = logging.FileHandler(log_file_path, mode='a')  # Always append after initial flush
            self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s' if show_timestamp else '%(message)s'))
            self.logger.addHandler(self.file_handler)
            vprint(f"### Log file is flushed (created) as {log_file_path} ###")
        else:
            self.file_handler = None
            vprint(f"### Log file is not flushed (created) because log_file is set to {self.log_file} ###")
        vprint(' ')
        
    def close(self):
        """Closes the file handler if it exists."""
        if self.file_handler is not None:
            self.file_handler.flush()
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
            self.file_handler = None

def print_system_info():
    
    import importlib
    import importlib.metadata
    import os
    import platform
    # import sys
    
    vprint("### System information ###")
    
    # Operating system information
    vprint(f"Operating System: {platform.system()} {platform.release()}")
    vprint(f"OS Version: {platform.version()}")
    vprint(f"Machine: {platform.machine()}")
    vprint(f"Processor: {platform.processor()}")
    
    # CPU cores
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        cpus =  int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        # Fallback to the total number of CPU cores on the node
        cpus = os.cpu_count()
    vprint(f"Available CPU cores: {cpus}")
    
    # Memory information
    if 'SLURM_MEM_PER_NODE' in os.environ:
        # Memory allocated per node by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_NODE']) / 1024  # Convert MB to GB
        vprint(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    elif 'SLURM_MEM_PER_CPU' in os.environ:
        # Memory allocated per CPU by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_CPU']) * cpus / 1024  # Convert MB to GB
        vprint(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    else:
        try:
            import psutil
            # Fallback to system memory information
            mem = psutil.virtual_memory()
            vprint(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
            vprint(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
        except ImportError:
            vprint("Memory information will be available after `conda install conda-forge::psutil`")
    try:
        import torch
        # GPU information
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            vprint(f"CUDA Available: {torch.cuda.is_available()}")
            vprint(f"CUDA Version: {torch.version.cuda}")
            vprint(f"Available CUDA GPUs: {[torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]}")
            vprint(f"CUDA Compute Capability: {[f'{major}.{minor}' for (major, minor) in [torch.cuda.get_device_capability(d) for d in range(torch.cuda.device_count())]]}")
            vprint("  INFO: For torch.compile with Triton, you'll need CUDA GPU with Compute Capability >= 7.0.")
            vprint("        In addition, Triton does not directly support Windows.")
            vprint("        For Windows users, please follow the instruction and download `triton-windows` from https://github.com/woct0rdho/triton-windows.")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            vprint(f"MPS Available: {torch.backends.mps.is_available()}")
        elif torch.backends.cuda.is_built() or torch.backends.mps.is_built():
            vprint("GPU support built with PyTorch, but could not find any GPU device.")
        else:
            vprint("No GPU backend (CUDA or MPS) built into this PyTorch install.")
            vprint("Install a PyTorch version with GPU support if you want to utilize GPUs.")
    except ImportError:
        vprint("No GPU information because PyTorch can't be imported")
    
    # Python version and executable
    # vprint(f"Python Executable: {sys.executable}")
    # vprint(f"Python Version: {sys.version}")
    
    # Print package versions
    packages = [
        ("Numpy", "numpy"),
        ("PyTorch", "torch"),
        ("Optuna", "optuna"),
        ("Accelerate", "accelerate"),
    ]

    # Check versions for relevant packages
    for display_name, module_name in packages:
        try:
            # Try to get the version from package metadata (installed version)
            version = importlib.metadata.version(module_name)
            vprint(f"{display_name} Version (metadata): {version}")
        except importlib.metadata.PackageNotFoundError:
            vprint(f"{display_name} not found in the environment.")
        except Exception as e:
            vprint(f"Error retrieving version for {display_name}: {e}")
    
    # Check the version and path of the used PtyRAD package
    # Note that we're focusing on the version/path of the actual imported PtyRAD.
    # If there are both an installed version of PtyRAD in the environment and a local copy in the working directory,
    # Python will prioritize the version in the working directory.
    #
    # When using `pip install -e .`, only the version metadata gets recorded, which won't be updated until you reinstall.
    # As a result, a user who pulls new code from the repo will have their `__init__.py` updated, but the version metadata recorded by pip will remain unchanged.
    # Therefore, it is better to retrieve the version directly from `module.__version__` for now, as this will reflect the actual local version being used.
    # Once we transition to using pip/conda for installation, all code updates will be paired with an installation, 
    # and we can safely switch to retrieving the version via `importlib.metadata.version`.
    try:
        # Import ptyrad (which will prioritize the local version if available)
        module = importlib.import_module('ptyrad')
        vprint(f"PtyRAD Version (direct import): {module.__version__}") # This version is defined in __init__.py
        vprint(f"PtyRAD is located at: {module.__file__}")
    except ImportError:
        vprint("PtyRAD not found locally")
    except AttributeError:
        vprint("PtyRAD imported, but no __version__ attribute found.")
    except Exception as e:
        vprint(f"Error retrieving version for PtyRAD: {e}")
    
    vprint(" ")

def set_gpu_device(gpuid=0):
    vprint("### Setting GPU Device ###")

    if gpuid is None:
        device = torch.device("cpu")
        vprint("Specified to use CPU (gpuid=None).")

    else:
        if torch.cuda.is_available():
            num_cuda_devices = torch.cuda.device_count()
            if gpuid < num_cuda_devices:
                device = torch.device(f"cuda:{gpuid}")
                torch.set_default_device(device)
                vprint(f"Selected GPU device: {device} ({torch.cuda.get_device_name(gpuid)})")
            else:
                device = torch.device("cuda")
                vprint(f"Requested CUDA device cuda:{gpuid} is out of range (only {num_cuda_devices} available)." 
                       f"Fall back to GPU device: {device}")
        
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            vprint("Selected GPU device: MPS (Apple Silicon)")
        
        else:
            device = torch.device("cpu")
            vprint(f"GPU ID specifed as {gpuid} but no GPU found. Using CPU instead.")
            
    torch.set_default_device(device)

    vprint(" ")
    return device

@torch.compiler.disable
def vprint(*args, verbose=True, **kwargs):
    """Verbose print/logging with individual control, only for rank 0 in DDP."""
    if verbose and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
        logger = logging.getLogger('PtyRAD')
        if logger.hasHandlers():
            logger.info(' '.join(map(str, args)), **kwargs)
        else:
            print(*args, **kwargs)

def vprint_nested_dict(d, indent=0, verbose=True, leaf_inline_threshold=6):
    indent_str = "    " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            # Check if this is a flat leaf dict
            is_flat_leaf = all(not isinstance(v, (dict, list)) for v in value.values())
            if is_flat_leaf and len(value) <= leaf_inline_threshold:  # Determine whether to print inline or not
                flat = ", ".join(f"{k}: {repr(v)}" for k, v in value.items())
                vprint(f"{indent_str}{key}: {{{flat}}}", verbose=verbose)
            else:
                vprint(f"{indent_str}{key}:", verbose=verbose)
                vprint_nested_dict(value, indent + 1, verbose=verbose)
        elif isinstance(value, list) and all(not isinstance(i, (dict, list)) for i in value):
            vprint(f"{indent_str}{key}: {value}", verbose=verbose)
        else:
            vprint(f"{indent_str}{key}: {repr(value)}", verbose=verbose)

def safe_get_nested(d, keys, default=None):
    """
    Safely get a value from a nested dictionary.
    
    Parameters:
    - d (dict): The dictionary to traverse.
    - keys (list or tuple): A sequence of keys to access nested values.
    - default: The value to return if any key is missing or intermediate value is None.
    
    Returns:
    - The nested value if found, otherwise `default`.
    """
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
        if d is None:
            return default
    return d

def get_date(date_format='%Y%m%d'):
    from datetime import date, datetime
    
    # If the format includes time-specific placeholders, return full datetime
    if any(fmt in date_format for fmt in ['%H', '%M', '%S']):
        return datetime.now().strftime(date_format)
    
    # Otherwise, just return the date
    return date.today().strftime(date_format)

@torch.compiler.disable
def time_sync():
    # PyTorch doesn't have a direct exposed API to check the selected default device 
    # so we'll be checking these .is_available() just to prevent error.
    # Luckily these checks won't really affect the performance.
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Check if MPS (Metal Performance Shaders) is available (macOS only)
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    # Measure the time
    t = perf_counter()
    return t

def parse_sec_to_time_str(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if days > 0:
        return f"{int(days)} day {int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif hours > 0:
        return f"{int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif minutes > 0:
        return f"{int(minutes)} min {secs:.3f} sec"
    else:
        return f"{secs:.3f} sec"

def parse_hypertune_params_to_str(hypertune_params):
    
    hypertune_str = ''
    for key, value in hypertune_params.items():
        if key[-2:].lower() == "lr":
            hypertune_str += f"_{key}_{value:.1e}"
        elif isinstance(value, (int, float)):
            hypertune_str += f"_{key}_{value:.3g}"
        else:
            hypertune_str += f"_{key}_{value}"
    
    return hypertune_str