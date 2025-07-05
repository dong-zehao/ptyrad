# Python script to run PtyRAD with command-line interface
# Updated by Chia-Hao Lee on 2025.05.12

# Before running this script, you must first follow the instruction in `README.md` to:
# 1. Create the Python environment with all dependant Python packages like PyTorch
# 2. Activate that Python environment
# 3. Install `ptyrad` package into your activated Python environement (only need to install once)

import argparse

from ptyrad.load import load_params
from ptyrad.reconstruction import PtyRADSolver
from ptyrad.utils import CustomLogger, print_system_info, set_accelerator, set_gpu_device

if __name__ == "__main__":
    # If you want to run from demo/ directory with GPU 0,
    # python ../scripts/run_ptyrad.py --params_path "params/tBL_WSe2_reconstruct.yml" --gpuid 0
    
    # If you want to run independent processes on 2 GPUs on your workstation from Bash terminal
    # parallel --delay 5 python ../scripts/run_ptyrad.py --params_path "params/tBL_WSe2_reconstruct.yml" --gpuid ::: 0 1
    
    parser = argparse.ArgumentParser(
        description="Run PtyRAD", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--params_path", type=str, required=True)
    parser.add_argument("--gpuid",       type=int, required=False, default=None)
    parser.add_argument("--jobid",       type=int, required=False, default=0)
    args = parser.parse_args()
    
    # Set up custom logger
    logger = CustomLogger(log_file='ptyrad_log.txt', log_dir='auto', prefix_date=True, prefix_jobid=args.jobid, append_to_file=True, show_timestamp=True)
    
    # Set up accelerator for multiGPU/mixed-precision setting, note that thess has no effect when we launch it with just `python <script>`
    # accelerator = set_accelerator()
    
    print_system_info()
    params = load_params(args.params_path)
    device = set_gpu_device(args.gpuid)
    
    ptycho_solver = PtyRADSolver(params, device=device, logger=logger)

    ptycho_solver.run()

