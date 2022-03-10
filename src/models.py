import numpy as np
import torch.nn as nn

def get_models(model, image_shape, latent_dim):

    class GAN_Generator(nn.Module):
        def __init__(self):
            super(GAN_Generator, self).__init__()

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(latent_dim, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(image_shape))),
                nn.Tanh()
            )

        def forward(self, z):
            img = self.model(z)
            img = img.view(img.size(0), *image_shape)
            return img


    class GAN_Discriminator(nn.Module):
        def __init__(self):
            super(GAN_Discriminator, self).__init__()

            self.model = nn.Sequential(
                nn.Linear(int(np.prod(image_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)

            return validity

    if model == 'gan':
        return GAN_Generator(), GAN_Discriminator()