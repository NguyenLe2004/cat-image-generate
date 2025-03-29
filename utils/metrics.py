import torch
import tqdm
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

def compute_gan_metrics(generator, dataloader, latent_dim):
    fid = FrechetInceptionDistance(normalize=True).to(generator.device)
    kid = KernelInceptionDistance(subset_size=50, normalize=True).to(generator.device)
    inception_score = InceptionScore(normalize=True).to(generator.device)

    num_batches = 0

    with torch.no_grad():
        for real_images in tqdm(dataloader, desc="Computing GAN Metrics"):
            real_images = real_images.to(generator.device)
            batch_size = real_images.size(0)
            fake_images = generator(torch.randn(batch_size, latent_dim, 1, 1, device=generator.device))  # Tạo ảnh giả

            # Cập nhật FID và KID
            fid.update(real_images, real=True)
            fid.update(fake_images, real=False)

            kid.update(real_images, real=True)
            kid.update(fake_images, real=False)

            inception_score.update(fake_images)
            num_batches += 1

    # Tính giá trị cuối cùng
    fid_score = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    kid_mean, kid_std = kid_mean.item(), kid_std.item()
    is_mean, is_std = inception_score.compute()
    is_mean, is_std = is_mean.item(), is_std.item()

    return {
        "FID": fid_score,
        "KID": kid_mean,
        "Inception Score": is_mean
    }