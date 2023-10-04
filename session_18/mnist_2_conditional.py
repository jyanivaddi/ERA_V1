# prerequisites
import random

import matplotlib.pyplot as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, num_labels=10, label_enc_dim=28):
        super(VAE, self).__init__()

        # label embedding
        self.label_enc_dim = label_enc_dim
        self.num_labels = num_labels
        self.class_embeddings = nn.Embedding(self.num_labels, self.label_enc_dim)

        # encoder part
        self.enc_inp_dim = x_dim+(self.num_labels*self.label_enc_dim)
        self.fc1 = nn.Linear(self.enc_inp_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.dec_inp_dim = z_dim + (self.num_labels*self.label_enc_dim)
        self.fc4 = nn.Linear(self.dec_inp_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)


    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h)  # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)  # return z sample

    def decoder(self, z, label):
        label_enc = self.get_label_embedding(label)
        z = torch.cat([z, label_enc], dim=1)
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h))

    def get_label_embedding(self, label):
        label_one_hot = torch.nn.functional.one_hot(label, num_classes=10)
        label_enc = self.class_embeddings(label_one_hot)
        label_enc = torch.flatten(label_enc, start_dim=1)
        return label_enc

    def forward(self, x, label):
        x = x.view(-1, 784)
        label_enc = self.get_label_embedding(label)
        x = torch.cat([x, label_enc], dim=1)
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        return self.decoder(z, label), mu, log_var


def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.cuda()
        labels = labels.cuda()
        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data, labels)
        loss = loss_function(recon_batch, data, mu, log_var)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def test(epoch):
    vae.eval()
    test_loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.cuda()
            labels = labels.cuda()
            recon, mu, log_var = vae(data, labels)

            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
            if epoch % 10 == 0:
                z = torch.randn(64, 2).cuda()
                this_label = 5 * torch.ones(64, dtype=torch.int64).cuda()
                sample = vae.decoder(z, this_label).cuda()
                save_image(sample.view(64, 1, 28, 28), f"./samples/sample_{epoch}.png")

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


def show_image(img_tensor):
    np_op_img = np.asarray(img_tensor.cpu()).squeeze()
    pl.figure()
    pl.imshow(np_op_img, cmap='gray')
    pl.colorbar()
    pl.show()


def conditional_generation(vae, x, label, weight=0.1):
    x = x.view(-1, 784)
    l_e = vae.get_label_embedding(label)
    x_l = torch.cat([x, l_e], dim=1)

    with torch.no_grad():
        # get mean and std of given image and label
        mu0, log_var0 = vae.encoder(x_l)
        z0 = vae.sampling(mu0, log_var0)

        # draw a random value of mean and std from standard gaussian
        #z1 = torch.randn(1, 2).cuda()
        img_tensor = vae.decoder(z0, label).view(1, 28, 28)
        show_image(img_tensor)
        #z_interp = torch.lerp(z0, z1, weight=weight)
        #misc_labels = np.random.choice(np.arange(10), size=64, replace=True)
        #gen_images = []
        #for tgt in misc_labels:
        #    tgt_label_tensor = torch.tensor(tgt,dtype=torch.int64).unsqueeze(0).cuda()
        #    img_tensor = vae.decoder(z_interp, tgt_label_tensor).view(1, 28, 28)
        #    gen_images.append(img_tensor)
        #final_output = torch.vstack(gen_images).unsqueeze(1)
        #save_image(final_output, f'./samples/final_output_{weight*100}p.png')
        return img_tensor


if __name__ == "__main__":
    bs = 64
    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True)

    # build model
    vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
    if torch.cuda.is_available():
        vae.cuda()

    optimizer = optim.Adam(vae.parameters())

    for epoch in range(1, 30):
        train(epoch)
        test(epoch)

    #with torch.no_grad():
        #for lbl in range(10):
         #   z = torch.randn(64, 2).cuda()
          #  label = lbl*torch.ones(64, dtype=torch.int64).cuda()
           # sample = vae.decoder(z, label).cuda()
#
 #           save_image(sample.view(64, 1, 28, 28), f'./samples/sample_{lbl}.png')

    with torch.no_grad():
        data, labels = next(iter(test_loader))
        data = data.cuda()
        labels = labels.cuda()
        op = conditional_generation(vae, data[0].unsqueeze(0), labels[0].unsqueeze(0))
