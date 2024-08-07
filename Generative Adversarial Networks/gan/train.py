from glob import glob
import os
import torch
from utils import get_fid, interpolate_latent_space, save_plot
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchvision.datasets import VisionDataset


def build_transforms():
    # TODO 1.2: Add two transforms:
    # 1. Convert input image to tensor.
    # 2. Rescale input image to be between -1 and 1.

    # https://stackoverflow.com/questions/58954799/how-to-normalize-pil-image-between-1-and-1-in-pytorch-for-transforms-compose
    ds_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) ])
    return ds_transforms


def get_optimizers_and_schedulers(gen, disc):
    # TODO 1.2 Get optimizers and learning rate schedulers.
    # 1. Construct the optimizers for the discriminator and generator.
    # Both should use the Adam optimizer with learning rate of .0002 and Beta1 = 0, Beta2 = 0.9.
    # 2. Construct the learning rate schedulers for the generator and discriminator.
    # The learning rate for the discriminator should be decayed to 0 over 500K steps.
    # The learning rate for the generator should be decayed to 0 over 100K steps.

    optim_discriminator = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0, 0.9))
    scheduler_discriminator = torch.optim.lr_scheduler.LinearLR(optim_discriminator,1,1/500000,500000)
    # scheduler_discriminator = torch.optim.lr_scheduler.PolynomialLR(optim_discriminator,total_iters = 500000,power=1.0)
    optim_generator = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0, 0.9))
    scheduler_generator = torch.optim.lr_scheduler.LinearLR(optim_generator,1,1/100000,100000)
    # scheduler_discriminator = torch.optim.lr_scheduler.PolynomialLR(optim_generator,total_iters = 100000,power=1.0)

    return (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    )


class Dataset(VisionDataset):
    def __init__(self, root, transform=None):
        super(Dataset, self).__init__(root)
        self.file_names = glob(os.path.join(self.root, "*.jpg"), recursive=True)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.file_names[index])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.file_names)


def train_model(
    gen,
    disc,
    num_iterations,
    batch_size,
    lamb=10,
    prefix=None,
    gen_loss_fn=None,
    disc_loss_fn=None,
    log_period=10000,
):
    torch.backends.cudnn.benchmark = True
    ds_transforms = build_transforms()
    train_loader = torch.utils.data.DataLoader(
        Dataset(root="datasets/CUB_200_2011_32", transform=ds_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    (
        optim_discriminator,
        scheduler_discriminator,
        optim_generator,
        scheduler_generator,
    ) = get_optimizers_and_schedulers(gen, disc)
    scaler = torch.cuda.amp.GradScaler()

    print("Length Trainloader = ", len(train_loader))
    iters = 0
    fids_list = []
    iters_list = []
    while iters < num_iterations:
        # print('#################################################### ', iters)
        for train_batch in train_loader:
            with torch.cuda.amp.autocast():
                train_batch = train_batch.cuda()
                # print("Train() : Train batch shape = ", train_batch.shape)
                # TODO 1.2: compute generator outputs and discriminator outputs
                # 1. Compute generator output -> the number of samples must match the batch size.
                
                # Cant use batch size for gen as last train_batch has != 256
                gen_out = gen(train_batch.shape[0])
                # 2. Compute discriminator output on the train batch.
                discrim_real = disc(train_batch)
                # 3. Compute the discriminator output on the generated data.
                discrim_fake = disc(gen_out)

                # TODO: 1.5 Compute the interpolated batch and run the discriminator on it.
                # To compute interpolated data, draw eps ~ Uniform(0, 1)
                # interpolated data = eps * fake_data + (1-eps) * real_data

                # Is eps singular value or matrix? Uniform not normal
                # eps = torch.normal(0,1,train_batch.shape)
                eps = torch.rand(1).cuda()
                # print("Eps / shape = ", eps, eps.shape)
                # print("Gen out/ train_batch shape = ", gen_out.shape,train_batch.shape)
                interp = eps * gen_out + (1 - eps) * train_batch
                discrim_interp = disc(interp)

                discriminator_loss = disc_loss_fn(
                    discrim_real, discrim_fake, discrim_interp, interp, lamb
                )
            optim_discriminator.zero_grad(set_to_none=True)
            scaler.scale(discriminator_loss).backward()
            scaler.step(optim_discriminator)
            scheduler_discriminator.step()

            if iters % 5 == 0:
                with torch.cuda.amp.autocast():
                    # TODO 1.2: Compute samples and evaluate under discriminator.
                    # Below two lines not necessary?
                    gen_out = gen(train_batch.shape[0])
                    discrim_fake = disc(gen_out)

                    generator_loss = gen_loss_fn(discrim_fake)
                optim_generator.zero_grad(set_to_none=True)
                scaler.scale(generator_loss).backward()
                scaler.step(optim_generator)
                scheduler_generator.step()

            if iters % log_period == 0 and iters != 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        # TODO 1.2: Generate samples using the generator, make sure they lie in the range [0, 1].
                        generated_samples = gen(train_batch.shape[0])
                        # Correct way to convert in range[0,1]?
                        generated_samples = (generated_samples+1) * 0.5
                        # generated_samples = torch.nn.Sigmoid(generated_samples)

                    save_image(
                        generated_samples.data.float(),
                        prefix + "samples_{}.png".format(iters),
                        nrow=10,
                    )
                    torch.jit.save(gen, prefix + "/generator.pt")
                    torch.jit.save(disc, prefix + "/discriminator.pt")
                    fid = fid = get_fid(
                        gen,
                        dataset_name="cub",
                        dataset_resolution=32,
                        z_dimension=128,
                        batch_size=256,
                        num_gen=10_000,
                    )
                    print(f"Iteration {iters} FID: {fid}")
                    fids_list.append(fid)
                    iters_list.append(iters)

                    save_plot(
                        iters_list,
                        fids_list,
                        xlabel="Iterations",
                        ylabel="FID",
                        title="FID vs Iterations",
                        filename=prefix + "fid_vs_iterations",
                    )
                    interpolate_latent_space(
                        gen, prefix + "interpolations_{}.png".format(iters)
                    )
            scaler.update()
            iters += 1
    fid = get_fid(
        gen,
        dataset_name="cub",
        dataset_resolution=32,
        z_dimension=128,
        batch_size=256,
        num_gen=50_000,
    )
    print(f"Final FID (Full 50K): {fid}")