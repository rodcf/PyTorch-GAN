import numpy as np
import torch
import torch.nn as nn

def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

def get_models(model, image_shape, latent_dim, n_classes=10):

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
                nn.Dropout(0.4),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        def forward(self, img):
            img_flat = img.view(img.size(0), -1)
            validity = self.model(img_flat)

            return validity

    class DCGAN_Generator(nn.Module):
        def __init__(self):
            super(DCGAN_Generator, self).__init__()
            ngf = image_shape[1]
            nc = image_shape[0]

            self.init_size = ngf // 4
            self.l1 = nn.Sequential(nn.Linear(latent_dim, ngf*4 * self.init_size ** 2))

            self.conv_blocks = nn.Sequential(
                nn.BatchNorm2d(ngf*4),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf*4, ngf*4, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf*4, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ngf*4, ngf*2, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf*2, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf*2, nc, 3, stride=1, padding=1),
                nn.Tanh(),
            )

        def forward(self, z):
            out = self.l1(z)
            out = out.view(out.shape[0], image_shape[1]*4, self.init_size, self.init_size)
            img = self.conv_blocks(out)
            return img

    class DCGAN_Discriminator(nn.Module):
        def __init__(self):
            super(DCGAN_Discriminator, self).__init__()
            ndf = image_shape[1]
            nc = image_shape[0]
            
            def discriminator_block(in_filters, out_filters, bn=True):
                block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
                if bn:
                    block.append(nn.BatchNorm2d(out_filters, 0.8))
                return block

            self.model = nn.Sequential(
                *discriminator_block(nc, ndf//2, bn=False),
                *discriminator_block(ndf//2, ndf),
                *discriminator_block(ndf, ndf*2),
                *discriminator_block(ndf*2, ndf*4),
            )

            # The height and width of downsampled image
            # ds_size = image_shape[1] // 2 ** 4
            self.adv_layer = nn.Sequential(nn.Linear(ndf*4 * 2 ** 2, 1), nn.Sigmoid())

        def forward(self, img):
            out = self.model(img)
            out = out.view(out.shape[0], -1)
            validity = self.adv_layer(out)

            return validity

    class CGAN_Generator(nn.Module):
        def __init__(self):
            super(CGAN_Generator, self).__init__()

            self.label_emb = nn.Embedding(n_classes, n_classes)

            def block(in_feat, out_feat, normalize=True):
                layers = [nn.Linear(in_feat, out_feat)]
                if normalize:
                    layers.append(nn.BatchNorm1d(out_feat, 0.8))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                return layers

            self.model = nn.Sequential(
                *block(latent_dim + n_classes, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, int(np.prod(image_shape))),
                nn.Tanh()
            )

        def forward(self, noise, labels):
            # Concatenate label embedding and image to produce input
            gen_input = torch.cat((self.label_emb(labels), noise), -1)
            img = self.model(gen_input)
            img = img.view(img.size(0), *image_shape)
            return img


    class CGAN_Discriminator(nn.Module):
        def __init__(self):
            super(CGAN_Discriminator, self).__init__()

            self.label_embedding = nn.Embedding(n_classes, n_classes)

            self.model = nn.Sequential(
                nn.Linear(n_classes + int(np.prod(image_shape)), 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 512),
                nn.Dropout(0.4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 1),
            )

        def forward(self, img, labels):
            # Concatenate label embedding and image to produce input
            d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
            validity = self.model(d_in)
            return validity


    if model.lower() == 'gan':
        return GAN_Generator(), GAN_Discriminator()
    elif model.lower() == 'dcgan':
        gen, disc = DCGAN_Generator(), DCGAN_Discriminator()
        return gen.apply(weights_init_normal), disc.apply(weights_init_normal)
    elif model.lower() == 'cgan':
        return CGAN_Generator(), CGAN_Discriminator()