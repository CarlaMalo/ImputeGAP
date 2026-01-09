import time

from imputegap.wrapper.AlgoPython.NuwaTS.llms_recovery import llms_recov
from imputegap.wrapper.AlgoPython.NuwaTS2.run_nuwats import run_nuwats

def nuwats(incomp_data, seq_length=-1, patch_size=-1, batch_size=-1, pred_length=-1, label_length=-1, enc_in=10, dec_in=10, c_out=10, gpt_layers=6, num_workers=0, tr_ratio=0.9, seed=42, logs=True, verbose=True):
    """
    Perform imputation using NuwaTS: Transformer-based recovery from missing values in multivariate time series.


    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    seq_length : int, optional
        Length of the input sequence for the encoder. If -1, it will be automatically determined (default: -1).

    patch_size : int, optional
        Patch size used for segmenting the sequence in the NuwaTS model (default: -1).

    batch_size : int, optional
        Number of samples per batch during training/inference. If -1, it will be auto-set (default: -1).

    pred_length : int, optional
        Length of the output prediction window (default: -1).

    label_length : int, optional
        Length of the label segment used during decoding (default: -1).

    enc_in : int, optional
        Number of input features for the encoder (default: 10).

    dec_in : int, optional
        Number of input features for the decoder (default: 10).

    c_out : int, optional
        Number of output features of the model (default: 10).

    gpt_layers : int, optional
        Number of layers in the transformer/generator component (default: 6).

    num_workers: int, optional
         Number of worker for multiprocess (default is 0).

    tr_ratio: float, optional
         Split ratio between training and testing sets (default is 0.9).

    seed : int, optional
        Random seed for reproducibility (default: 42).

    logs : bool, optional
        Whether to print/log execution time and key events (default: True).

    verbose : bool, optional
        Whether to print detailed output information during execution (default: True).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values filled in.

    Example
    -------
        >>> imputed = nuwats(incomp_data, seq_length=48, batch_size=16, patch_size=4)
        >>> print(imputed.shape)

    References
    ----------
    Cheng, Jinguo and Yang, Chunwei and Cai, Wanlin and Liang, Yuxuan and Wen, Qingsong and Wu, Yuankai: "NuwaTS: Mending Every Incomplete Time Series", arXiv'2024
    https://github.com/Chengyui/NuwaTS/tree/master
    """
    start_time = time.time()  # Record start time

    recov_data = llms_recov(ts_m=incomp_data, seq_length=seq_length, patch_size=patch_size, batch_size=batch_size, pred_length=pred_length, label_length=label_length, enc_in=enc_in, dec_in=dec_in, c_out=c_out, gpt_layers=gpt_layers, num_workers=num_workers, tr_ratio=tr_ratio, model="NuwaTS", seed=seed, verbose=verbose)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation nuwats - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data

def nuwats2(incomp_data, seq_length=-1, patch_size=-1, batch_size=-1, gpt_layers=6, num_workers=0, tr_ratio=0.9, seed=42, logs=True, verbose=True, original_tr_mask=False, data_name="custom"):
    """
    Perform imputation using NuwaTS: Transformer-based recovery from missing values in multivariate time series.


    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input matrix with contamination (missing values represented as NaNs).

    seq_length : int, optional
        Length of the input sequence for the encoder. If -1, it will be automatically determined (default: -1).

    patch_size : int, optional
        Patch size used for segmenting the sequence in the NuwaTS model (default: -1).

    batch_size : int, optional
        Number of samples per batch during training/inference. If -1, it will be auto-set (default: -1).

    gpt_layers : int, optional
        Number of layers in the transformer/generator component (default: 6).

    num_workers: int, optional
         Number of worker for multiprocess (default is 0).

    tr_ratio: float, optional
         Split ratio between training and testing sets (default is 0.9).

    seed : int, optional
        Random seed for reproducibility (default: 42).

    logs : bool, optional
        Whether to print/log execution time and key events (default: True).

    verbose : bool, optional
        Whether to print detailed output information during execution (default: True).

    original_tr_mask : bool, optional
        Whether to use the original training mask during imputation (default: False).
    
    data_name : str, optional
        Name of the dataset being processed (default: custom).

    Returns
    -------
    numpy.ndarray
        The imputed matrix with missing values filled in.

    Example
    -------
        >>> imputed = nuwats(incomp_data, seq_length=48, batch_size=16, patch_size=4)
        >>> print(imputed.shape)
    
    Checkpoint 
    -----------
    If a checkpoint for the specified configuration exists, the function will automatically load it to skip training.
    Location: ./imputegap_assets/models/checkpoints/NuwaTS_{data_name}_finetuned/checkpoint.pth

    References
    ----------
    Cheng, Jinguo and Yang, Chunwei and Cai, Wanlin and Liang, Yuxuan and Wen, Qingsong and Wu, Yuankai: "NuwaTS: Mending Every Incomplete Time Series", arXiv'2024
    https://github.com/Chengyui/NuwaTS/tree/master
    """
    start_time = time.time()  # Record start time

    recov_data = run_nuwats(ts_m=incomp_data, seq_length=seq_length, patch_size=patch_size, batch_size=batch_size, gpt_layers=gpt_layers, num_workers=num_workers, tr_ratio=tr_ratio, model="NuwaTS", seed=seed, verbose=verbose, original_tr_mask=original_tr_mask, data_name=data_name)

    end_time = time.time()
    if logs and verbose:
        print(f"\n> logs: imputation nuwats - Execution Time: {(end_time - start_time):.4f} seconds\n")

    return recov_data

