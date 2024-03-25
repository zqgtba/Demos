from skimage.metrics import peak_signal_noise_ratio, structural_similarity, normalized_root_mse
import fastmri

def calculate_psnr_fastmri(gt, pred):
    """Compute Peak Signal Noise Ratio metric (PSNR)"""
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max()) # gt.max()

def calculate_ssim_fastmri(gt, pred):
    """Compute structural similarity metric (SSIM)"""
    return structural_similarity(gt, pred, data_range=gt.max()) # gt.max()

def calculate_nrmse_fastmri(gt, pred):
    """Compute normalized root mean-squared error (NRMSE)"""
    return normalized_root_mse(gt, pred)

# check if the data need to be normlized!

label = fastmri.complex_abs(label).cpu().detach().numpy()
pred = fastmri.complex_abs(pred).cpu().detach().numpy()

psnr = calculate_psnr_fastmri(label, pred)
ssim = calculate_ssim_fastmri(label, pred)
nrmse = calculate_nrmse_fastmri(label, pred)
