import os

import h5py
import numpy as np
import scipy.io as sio

from ptyrad.utils import vprint


###### These are data loading functions ######

def load_raw(file_path, shape, dtype=np.float32, offset=0, gap=1024):
    # shape = (N, height, width)
    # np.fromfile with custom dtype is faster than the np.read and np.frombuffer
    # This implementaiton is also roughly 2x faster (10sec vs 20sec) than load_hdf5 with a 128x128x128x128 (1GB) EMPAD dataset
    # Note that for custom processed empad2 raw there might be no gap between the images
    N, height, width = shape

    # Verify file size first
    expected_size = offset + N * (height * width * dtype().itemsize + gap)
    actual_size = os.path.getsize(file_path)

    if actual_size != expected_size:
        raise ValueError(f"Mismatch in expected ({expected_size} bytes = offset + N * (height * width * 4 + gap)) vs. actual ({actual_size} bytes) file size! Check your loading configurations!")
    
    # Define the custom dtype to include both data and gap
    custom_dtype = np.dtype([
        ('data', dtype, (height, width)),
        ('gap', np.uint8, gap)  # uint8 means 1 byte per gap element
    ])

    # Read the entire file using the custom dtype
    with open(file_path, 'rb') as f:
        f.seek(offset)
        raw_data = np.fromfile(f, dtype=custom_dtype, count=N)

    # Extract just the 'data' part (ignoring the gaps)
    data = raw_data['data']
    vprint("Success! Loaded .raw file path =", file_path)
    vprint("Imported .raw data shape =", data.shape)
    vprint("Imported .raw data type =", data.dtype)
    return data

def load_tif(file_path):
    from tifffile import imread

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    data = imread(file_path)
    vprint("Success! Loaded .tif file path =", file_path)
    vprint("Imported .tif data shape =", data.shape)
    return data

def load_npy(file_path):

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    data = np.load(file_path)
    vprint("Success! Loaded .npy file path =", file_path)
    vprint("Imported .npy data shape =", data.shape)
    return data

###### These are reconstruction file loading functions ######
# Note that .mat and .hdf5 are also used for normal data

def load_fields_from_mat(file_path, target_field="All", squeeze_me=True, simplify_cells=True):
    """
    Load and extract specified fields from a MATLAB .mat file.

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html

    Parameters:
        file_path (str): The path to the MATLAB .mat file to be loaded and processed.
        target_field (str or list of str): The target field name(s) to extract from the .mat file.
            Specify a single field name as a string or multiple field names as a list of strings.
            Use "All" to load the entire .mat file.

    Returns:
        result_list (list or dict): A list containing the extracted field(s) as elements.
            If target_field is "All," the entire .mat file is returned as a dictionary.

    Raises:
        ValueError: If the nesting depth of target_field exceeds the maximum supported depth of 3.
        ValueError: If target_field is neither a string nor a list of strings.

    Examples:
        # Load the entire .mat file as a dictionary
        file_path = "your_file.mat"
        target_field = "All"
        result = load_fields_from_mat(file_path, target_field)

        # Extract a single field
        file_path = "your_file.mat"
        target_field = "object.sub_field"
        result = load_fields_from_mat(file_path, target_field)

        # Extract multiple fields
        file_path = "your_file.mat"
        target_field = ["object.sub_field", "another_object.field"]
        results = load_fields_from_mat(file_path, target_field)

        # Process the results
        for i, result in enumerate(results):
            if result is not None:
                vprint(f"Result {i + 1}: {result}")
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    result_list = []

    # Load entire .mat
    if target_field == "All":
        try:
            mat_contents = sio.loadmat(
                file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells
            )
            vprint("Success! Loaded .mat File path =", file_path)
            return mat_contents
        except NotImplementedError:
            # If loading from MATLAB file complains, switch to HDF5
            vprint("Can't load .mat v7.3 with 'scipy.io.loadmat'. Switching to h5py.")
            mat_contents = {}
            with h5py.File(file_path, "r") as hdf_file:
                for key in hdf_file.keys():
                    mat_contents[key] = hdf_file[key][()]
            vprint("Success! Loaded .mat file path =", file_path)
            return mat_contents

    # Check target_field type
    if isinstance(target_field, str):
        target_fields = [target_field]
    elif isinstance(target_field, list):
        target_fields = target_field
    else:
        raise ValueError("target_field must be a string or a list of strings")

    # Load field by field in target_fields (list)
    for name in target_fields:
        try:
            mat_contents = sio.loadmat(
                file_path, squeeze_me=squeeze_me, simplify_cells=simplify_cells
            )
            fields = name.split(".")
            outputs = mat_contents

            if len(fields) > 3:
                raise ValueError("The maximum supported nesting depth is 3.")

            for field in fields:
                if field in outputs:
                    if isinstance(outputs, sio.matlab.mio5.mat_struct):
                        outputs = getattr(outputs, field)
                    else:
                        outputs = outputs[field]
                else:
                    vprint(f"Field '{field}' not found in file {file_path}")
                    result_list.append(None)
                    break
            else:
                result_list.append(outputs)
        except NotImplementedError:
            # If loading from MATLAB file complains, switch to HDF5
            vprint("Can't load .mat v7.3 with scipy. Switching to h5py.")
            if name == 'outputs.probe_positions': # Convert the scipy syntax to hdf5 syntax
                name = 'outputs/probe_positions'
            data = load_hdf5(file_path, name)
            result_list.append(data)
    vprint("Success! Loaded .mat file path =", file_path)
    return result_list

def load_hdf5(file_path, dataset_key="ds"):
    """
    Load data from an HDF5 file.
    
    Parameters:
    
    file_path (str): The full path to the HDF5 data file.
    dataset_key (str, optional): The key of the dataset to load from the HDF5 file.
    
    Returns:
    data (numpy.ndarray): The loaded data.
    
    Raises:
    FileNotFoundError: If the specified file does not exist.
    
    Example:
    file_path = 'data.h5'
    data, data_source = load_hdf5(file_path, dataset_key='ds')
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")

    with h5py.File(file_path, "r") as hf:
        if dataset_key is None:
            vprint("Imported entire .hdf5 as a dict:", file_path)
            f = dict()
            for key in hf.keys():
                data = np.array(hf[key])
                if data.dtype == [('real', '<f8'), ('imag', '<f8')]: # For mat v7.3, the complex128 is read as this complicated datatype via h5py
                    vprint(f"Loaded data.dtype = {data.dtype}, cast it to 'complex128'")
                    data = data.view('complex128')
                f[key] = data
            vprint("Success! Loaded .hdf5 file path =", file_path)
            return f
            
        else:
            data = np.array(hf[dataset_key])
            if data.dtype == [('real', '<f8'), ('imag', '<f8')]: # For mat v7.3, the complex128 is read as this complicated datatype via h5py
                vprint(f"Loaded data.dtype = {data.dtype}, cast it to 'complex128'")
                data = data.view('complex128')
            vprint("Success! Loaded .hdf5 file path =", file_path)
            vprint("Imported .hdf5 data shape =", data.shape)
            vprint("Imported .hdf5 data type =", data.dtype)
            return data

def load_pt(file_path, weights_only=False):
    import torch

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")

    data = torch.load(file_path, weights_only=weights_only, map_location='cpu')
    # The default behavior of torch.load is `weights_only=True` since PyTorch 2.6 (2025.01.29)
    # https://dev-discuss.pytorch.org/t/bc-breaking-change-torch-load-is-being-flipped-to-use-weights-only-true-by-default-in-the-nightlies-after-137602/2573
    # Because PtyRAD .pt isn't a true PyTorch model, so `weights_only=True` would break this critical loading function.
    # However, `weights_only=False` has potential risk if the .pt file contains malicious code, so please only use this `load_pt` for PtyRAD-generated .pt file.
    
    vprint("Success! Loaded .pt file path =", file_path)
    return data

###### These are params loading functions ######

def load_params(file_path):
    
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file '{file_path}' does not exist.")
    
    vprint("### Loading params file ###")
    param_path, param_type = os.path.splitext(file_path)
    if param_type in (".yml", ".yaml"):
        params_dict = load_yml_params(file_path)
    elif param_type == ".toml":
        params_dict = load_toml_params(file_path)
    elif param_type == ".json":
        params_dict = load_json_params(file_path)
    elif param_type == ".py":
        params_dict =  load_py_params(param_path)
    else:
        raise ValueError("param_type needs to be either 'yml', 'json', or 'py'")
    
    # Add the file path to the params_dict while we save the params file to output folder
    params_dict['params_path'] = file_path
    
    vprint(" ")
    return params_dict

def load_json_params(file_path):
    import json
    
    with open(file_path, "r", encoding='utf-8') as file:
        params_dict = json.load(file)
    vprint("Success! Loaded .json file path =", file_path)
    return params_dict

def load_toml_params(file_path):
    """
    Load parameters from a TOML file.
    
    Parameters:
    file_path (str): The path to the TOML file to be loaded.
    
    Returns:
    dict: A dictionary containing the parameters loaded from the TOML file.
    
    Raises:
    FileNotFoundError: If the specified file does not exist.
    ImportError: If the tomli package is not installed for Python < 3.11.
    """

    try:
        # Read the file with utf-8
        # Note that "A TOML file must be a valid UTF-8 encoded Unicode document." per documentation.
        # Therefore, the toml file is read in binary mode ("rb") and the encoding is handled internally.
        # But I've observed some encoding mismatch when people run the script with terminal that has different default encoding.
        # Therefore, it is safer to read it with utf-8 encoding first and pass it to tomllib.
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read()
        
        try:
            # For Python 3.11+
            import tomllib
            params_dict = tomllib.loads(content)
        except ImportError:
            # For Python < 3.11
            import tomli # type: ignore
            params_dict = tomli.loads(content)
    except ImportError:
        raise ImportError("TOML support requires 'tomli' package for Python < 3.11 or built-in 'tomllib' for Python 3.11+. ")
    
    vprint("Success! Loaded .toml file path =", file_path)
    return params_dict

def load_yml_params(file_path):
    import yaml

    with open(file_path, "r", encoding='utf-8') as file:
        params_dict = yaml.safe_load(file)
    vprint("Success! Loaded .yml file path =", file_path)
    return params_dict

def load_py_params(file_path):
    import importlib

    params_module = importlib.import_module(file_path)
    params_dict = {
        name: getattr(params_module, name)
        for name in dir(params_module)
        if not name.startswith("__")
    }
    vprint("Success! Loaded .py file path =", file_path)
    return params_dict