import numpy as np
from tqdm import tqdm
import torch
import pytorch_lightning as pl
from torch import nn
from torchvision import models
import torch.optim as optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision import datasets
from pytorch_lightning.callbacks import Callback
from pl_bolts.datamodules import MNISTDataModule
from matplotlib.pyplot import imshow, figure, show
from torchvision.utils import make_grid
import numpy as np


class MNIST_Classifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, output_dim=10):
        super().__init__()
        self.fc_1 = nn.Sequential(nn.Linear(input_dim, hidden_dim, bias=False),
                                  nn.Dropout(0.1),
                                  nn.ReLU())
        self.fc_2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False),
                                  nn.Dropout(0.1),
                                  nn.ReLU())
        self.fc_3 = nn.Sequential(nn.Linear(hidden_dim, output_dim, bias=False),
                                  nn.ReLU())

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        return F.log_softmax(x, dim=1)


class MNIST_Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, encoding_dim=512):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc_3 = nn.Linear(hidden_dim, encoding_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        return x


class MNIST_Decoder(nn.Module):
    def __init__(self, output_dim=784, hidden_dim=512, latent_dim=256):
        super().__init__()
        self.fc_1 = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc_3 = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        return x


class PeriodicInference(Callback):
    def __init__(self, data_module):
        super().__init__()
        self.data_module = data_module
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def on_train_epoch_end(self, trainer, pl_module):
        print(f"classifier loss:{trainer.model.classifier_loss}")
        if (trainer.current_epoch +1) % 5 == 0:
            # plot SAMPLE IMAGES
            test_dl = self.data_module.val_dataloader()
            images, labels = next(iter(test_dl))
            one_image = images[0].unsqueeze(0)
            one_label = labels[0].unsqueeze(0)
            generate_images(trainer.model, one_image.to(self.device), one_label.to(self.device), self.device)


class ConditionalVAE(pl.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=28, num_classes=10, adversarial_prob=0.5):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = MNIST_Encoder()
        self.decoder = MNIST_Decoder()
        self.num_classes = num_classes
        self.adversarial_prob = adversarial_prob

        # class embeddings
        # self.class_embeddings = nn.Sequential(nn.Linear(num_classes, enc_out_dim),
        #                                      nn.ReLU())
        self.class_embeddings = nn.Embedding(num_classes, enc_out_dim)

        # distribution parameters
        self.fc_mu = nn.Sequential(nn.Linear(enc_out_dim, latent_dim), nn.ReLU())
        #self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        # mnist classifier
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.mnist_classifier = model.to(device)
        # self.mnist_loss_fn = F.nll_loss
        # self.classifier_loss = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        # print(f"log_pxz: {log_pxz.shape}")
        return log_pxz.sum(dim=1)


    def bernouli_loss(self, x_hat, x):
        #dist = torch.distributions.Bernoulli(x_hat)
        #log_pxz = dist.log_prob(x.type(torch.bool))
        #return log_pxz.sum(dim=1)
        return nn.functional.cross_entropy(x_hat, x)

    def kl_divergence_bernouli(self, z, bernouli_prob):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Bernoulli(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Bernoulli(bernouli_prob)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl



    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl



    def forward(self, x, label):
        # encode x to get the mu and variance parameters
        # print(x.shape)
        x_encoded = self.encoder(x)

        # Add label embedding to the image embeddings
        # c = self.class_embeddings(label)
        c = 0
        x_encoded = x_encoded + c

        #mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        bernouli_prob = self.fc_mu(x_encoded)

        # sample z from q
        #std = torch.exp(log_var / 2)
        #q = torch.distributions.Normal(mu, std)
        #q = torch.distributions.Bernoulli(bernouli_prob)
        #z = q.rsample()
        z = torch.bernoulli(bernouli_prob)

        # decoded
        x_hat = self.decoder(z)
        return x_hat, z, bernouli_prob


    def get_classifier_loss(self, x, labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mnist_classifier.eval()
        with torch.no_grad():
            output = self.mnist_classifier(x)
            loss = self.mnist_loss_fn(output.to(device), labels.to(device))
        return loss.item()

    def training_step(self, batch, batch_idx):
        x, label = batch
        x = torch.flatten(x, start_dim=1)

        # randomly shuffle labels
        if np.random.random() > 1. - self.adversarial_prob:
            # With view
            idx = torch.randperm(label.nelement())
            label = label.view(-1)[idx].view(label.size())

        # get classifier loss
        # self.classifier_loss = self.get_classifier_loss(x, label)
        self.classifier_loss = None

        #x_hat, mu, std, z = self(x, label)
        x_hat, z, bernouli_prob = self(x, label)
        print(x_hat.shape)

        # reconstruction loss
        #recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        recon_loss = self.bernouli_loss(x_hat, x)

        # kl
        #kl = self.kl_divergence(z, mu, std)
        kl = self.kl_divergence(z, bernouli_prob)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        if self.classifier_loss:
            final_loss = 0.5 * elbo + 0.5 * self.classifier_loss
        else:
            final_loss = elbo

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return final_loss


def model_train(model, device, train_loader, optimizer, train_acc, train_losses):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        data = torch.flatten(data, start_dim=1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
        correct+= output.argmax(dim=1).eq(target).sum().item()
        processed+= len(data)
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy = {100*correct/processed:0.2f}')

    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
    return  loss.item()

def train_mnist_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation([-15., 15.]),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    #test_transforms = transforms.Compose([
    #    transforms.ToTensor(),
    #    transforms.Normalize((0.1307,), (0.3081,))
    #])
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    #test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
    torch.manual_seed(1)
    batch_size = 128
    kwargs = {'num_workers': 2, 'pin_memory': True} if torch.cuda.is_available() else {}
    #train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    #test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)
    model = MNIST_Classifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, verbose=True)
    #train_losses = []
    #test_losses = []
    #train_acc = []
    #test_acc = []
    for epoch in range(1, 5):
        print(f"epoch: {epoch}")
        #this_train_loss = model_train(model, device, train_loader, optimizer, train_acc, train_losses)
        scheduler.step()
    #torch.save(model.state_dict(), 'classifier_model.pth')
    ## torch.save(model.state_dict(), 'classifier_model.pth')
    #model = MNIST_Classifier()
    #model.load_state_dict(torch.load('classifier_model.pth'))
    #model.eval()
    return model


def generate_images(model, image, label, device, num_images=32):
    predictions = []
    figure(figsize=(8, 3), dpi=300)

    with torch.no_grad():
        for cnt in range(num_images):
            num_preds = 1
            p = torch.distributions.Normal(torch.zeros([1, 256]), torch.ones([1, 256]))
            z = p.rsample((num_preds,))
            this_pred = model.decoder(z.to(model.device)).cpu()
            this_pred = this_pred.view(1, 28, 28).tile((3, 1, 1))
            # print(this_pred.max())
            # print(this_pred.min())
            predictions.append((this_pred * 255.0).type(torch.uint8))
    img = make_grid(torch.stack(predictions)).permute(1, 2, 0).cpu()
    # print(img.max())
    # print(img.min())
    # PLOT IMAGES
    imshow(img, cmap='gray_r')
    show()



if __name__ == "__main__":
    pl.seed_everything(1234)
    datamodule = MNISTDataModule('.')
    cond_vae = ConditionalVAE()
    trainer = pl.Trainer(gpus=1,
                         max_epochs=20,
                         callbacks=[PeriodicInference(data_module=datamodule)])
    trainer.fit(cond_vae, datamodule)
    #for cnt in range(1):
    #    one_image = imgs[cnt].unsqueeze(0)
    ##    one_label = torch.tensor([cnt]).unsqueeze(0)
    #    generate_images(cond_vae, one_image, one_label, cond_vae.device, num_images=32)

