import matplotlib.pyplot as plt

mask_ratios = [10, 20, 30, 40, 50, 60]

mae_muralxgan      = [ 0.0212,  0.0252,  0.0314, 0.0389, 0.0422, 0.0501]
mae_sota           = [ 0.0239,  0.0249,  0.0308, 0.0376, 0.0403, 0.0477]
mae_baseline       = [ 0.0249,  0.0288,  0.0351, 0.0413, 0.0488, 0.0578]


psnr_muralxgan   = [33.5, 31.2, 29.4, 25.3, 23.1, 19.6]
psnr_baseline    = [29.1, 28.2, 25.2, 22.1, 20.3, 15.4]
psnr_sota        = [31.9, 31.3, 29.7, 26.7, 24.9, 20.3]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=100)

axes[0].plot(mask_ratios, mae_muralxgan,    marker='o', label='MuralXGAN')
axes[0].plot(mask_ratios, mae_sota,            marker='s', label='SOTA')
# axes[0].plot(mask_ratios, fid_deepfillv2,     marker='x', label='Deepfillv2')
axes[0].plot(mask_ratios, mae_baseline,            marker='^', label='Without Text & Attention')


axes[0].set_xlabel('Mask Ratio (%)')
axes[0].set_ylabel('MAE')
axes[0].set_title('MAE')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)


axes[1].plot(mask_ratios, psnr_muralxgan,   marker='o', label='MuralXGAN')
axes[1].plot(mask_ratios, psnr_sota,           marker='s', label='SOTA')
# axes[1].plot(mask_ratios, psnr_deepfillv2,    marker='x', label='Deepfillv2')
axes[1].plot(mask_ratios, psnr_baseline,          marker='>', label='Without Text & Attention')


axes[1].set_xlabel('Mask Ratio (%)')
axes[1].set_ylabel('PSNR')
axes[1].set_title('PSNR')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)
# axes[1].set_ylim(15, 35)

plt.tight_layout()
plt.show()
