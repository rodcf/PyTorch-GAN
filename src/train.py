import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch
import json

from src.models import get_models

from time import perf_counter
from datetime import timedelta

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gan", help="which model to train. options are 'gan', 'cgan' or 'dcgan'")
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--mnist_location", type=str, default="data/mnist", help="download location of mnist images")
parser.add_argument("--results_location", type=str, default="results/", help="directory in which image samples and weights are stored")
parser.add_argument("--checkpoints_interval", type=int, default=20, help="how often to save the images and weights")
opt = parser.parse_args()
print(opt)

results_location = f"{opt.results_location}/{opt.model}"

os.makedirs(f"{results_location}/images", exist_ok=True)

os.makedirs(f"{results_location}/weights", exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss function
if opt.model == 'cgan':
    adversarial_loss = torch.nn.MSELoss()
else:
    adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator, discriminator = get_models(opt.model, img_shape, opt.latent_dim, opt.n_classes)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs(opt.mnist_location, exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST(
        opt.mnist_location,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def sample_image_cgan(n_row, epoch):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, f"{results_location}/images/{epoch+1}.png", nrow=n_row, normalize=True)

def sample_image(n_row, epoch):
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    gen_imgs = generator(z)
    save_image(gen_imgs.data, f"{results_location}/images/{epoch+1}.png", nrow=n_row, normalize=True)

# ----------
#  Training
# ----------

t1_start = perf_counter()

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        if opt.model == 'cgan':
            labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train GAN_Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        if opt.model == 'cgan':
            gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
        # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            validity = discriminator(gen_imgs, gen_labels)
        else:
            gen_imgs = generator(z)
            validity = discriminator(gen_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train GAN_Discriminator
        # ---------------------

        optimizer_D.zero_grad()


        validity_fake = discriminator(gen_imgs.detach(), gen_labels) if opt.model=='cgan' else discriminator(gen_imgs.detach())
            
        # Measure discriminator's ability to classify real from generated samples
        validity_real = discriminator(real_imgs, labels) if opt.model=='cgan' else discriminator(real_imgs)
        real_loss = adversarial_loss(validity_real, valid)

        validity_fake = discriminator(gen_imgs.detach(), gen_labels) if opt.model=='cgan' else discriminator(gen_imgs.detach())
        fake_loss = adversarial_loss(validity_fake, fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        if ((epoch + 1) % opt.checkpoints_interval == 0 or (epoch + 1) % opt.n_epochs == 0) and (i+1) == len(dataloader):
            if opt.model == 'cgan':
                args = (z,labels)
                input_names = ['input', 'labels']
            else:
                args = z
                input_names = ['input']
            if epoch + 1 == opt.n_epochs:
                torch.save(generator.state_dict(), f"{results_location}/weights/last.pth")
                torch.onnx.export(
                                    generator, args, 
                                    f"{results_location}/weights/last.onnx", 
                                    verbose=True, 
                                    input_names=input_names, 
                                    output_names = ['output'],
                                    export_params=True,
                                    dynamic_axes={
                                        'input': {0:'batch_size'},
                                        'labels': {0:'batch_size'},
                                        'output': {0:'batch_size'}
                                    }
                                )
            else:
                torch.save(generator.state_dict(), f"{results_location}/weights/{epoch+1}.pth")
                torch.onnx.export(
                                    generator, args, 
                                    f"{results_location}/weights/{epoch+1}.onnx", 
                                    verbose=True, 
                                    input_names=input_names, 
                                    output_names = ['output'],
                                    export_params=True,
                                    dynamic_axes={
                                        'input': {0:'batch_size'},
                                        'labels': {0:'batch_size'},
                                        'output': {0:'batch_size'}
                                    }
                                )
            if opt.model =='cgan':
                sample_image_cgan(n_row=10, epoch=epoch)
            else:
                sample_image(n_row=10, epoch=epoch)

t1_stop = perf_counter()

time_taken = str(timedelta(seconds=t1_stop-t1_start))

print("Time taken during training:", time_taken)

time = {}
time[f"{opt.model}_time"] = time_taken

with open(f"{opt.model}_time.json", 'w+') as metrics_file:
  json.dump(time, metrics_file)