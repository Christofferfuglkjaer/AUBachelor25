# "C:\Users\malth\OneDrive - Aarhus universitet\6. Semester\Bachelor projekt\AUBachelor25\Kode\Pytorch_AE.py"

import numpy as np
from medmnist import PneumoniaMNIST
import torch
import random
import copy
import time
from pytorch_msssim import ms_ssim

# Number of clients
n = input("Enter the number of clients: ")
n = int(n)
m = n

# Number of epochs
epochs = input("Enter the number of epochs: ")
epochs = int(epochs)

# How is it split
split = input("Enter the type of split (federated, full, limited): ")

# Load model
path = input("Enter the path to the model (\"\" for new model, \"Test\" for no model): ")

import neptune
run = neptune.init_run(project='momkeybomkey/Federated',
                       api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjNDdlN2ZhNy00ZmJmLTQ4YjMtYTk0YS1lNmViZmZjZWRhNzUifQ==',
                       tags=[f'split-{n}', split])

# Check for GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('', 'Using GPU', '')
else:
    device = torch.device('cpu')
    print('', 'Using CPU', '')

# Functions to load data
def _collate_fn(data):
        xs = []

        for x, _ in data:
            x = np.array(x).astype(np.float32) / 255.
            xs.append([x])

        return np.array(xs)

def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)
    total_size = len(index)
    i = 0
    random.shuffle(index)
    result = []

    while len(result) < total_size:
        result.append(index[i])
        i += 1
        
        if i >= total_size:
            i = 0
            random.shuffle(index)

    return result

def get_loader(dataset, random):
    total_size = len(dataset)
    # print('Size', total_size)
    if random:
        index_generator = shuffle_iterator(range(total_size))
    else:
        index_generator = list(range(total_size))

    while True:
        data = []

        for _ in range(len(index_generator)):
            idx = index_generator.pop()
            data.append(dataset[idx])

        return _collate_fn(data)

# Function to generate noise
def generate_noise(data, noise_factor = 0.08):
    data += noise_factor * (np.random.normal(size=data.shape) + np.random.poisson(data) / 255)
    data = np.clip(data, 0., 1.)
    return data

# Function to split data
def even_data(split):
    min_size = min([len(data) for data in split])
    return [data[:min_size] for data in split]

def split_data(data, n, noise_factor = 0.08):
    m = len(data)
    data_copy = copy.deepcopy(data)
    clean_split = []
    # Split the data mostly evenly
    for i in range(n):
        clean_split.append(data_copy[m * i // n: m * (i + 1) // n])
    # Even them out
    clean_split = even_data(clean_split)
    # Generate noisy data
    noisy_split = copy.deepcopy(clean_split)
    noisy_split = [generate_noise(x, noise_factor) for x in noisy_split]
    # Clip values
    clean_split = np.clip(np.array(clean_split), 0., 1.)
    noisy_split = np.clip(np.array(noisy_split), 0., 1.)
    # Convert to tensor
    clean_split = torch.tensor(clean_split).float()
    noisy_split = torch.tensor(noisy_split).float()
    # Send to device
    clean_split = clean_split.to(device)
    noisy_split = noisy_split.to(device)

    return clean_split, noisy_split

# Download data
train = PneumoniaMNIST(split="train", download=True, size=224)
test = PneumoniaMNIST(split="test", download=True, size=224)
val = PneumoniaMNIST(split="val", download=True, size=224)
# Load data
train_loader = get_loader(train, random=True)
test_loader = get_loader(test, random=True)

val_loader = get_loader(val, random=False)
# Split validation data
val_clean, val_noisy = split_data(val_loader, 1)
# Loss function
def loss_fn(recon_x, x, alpha = 0.84):
    L1Loss = torch.nn.L1Loss()
    L1 = L1Loss(recon_x, x)
    MS_SSIM = 1 - ms_ssim(recon_x.unsqueeze(0), x.unsqueeze(0), data_range=1, size_average=True)
    return alpha * MS_SSIM + (1 - alpha) * L1
# Function for PSNR
def PSNR(y_true, y_pred):
    mse_loss = torch.nn.MSELoss()
    return 20 * torch.log10(torch.max(y_true) / mse_loss(y_true, y_pred))
# Log during traning
def neptune_log(epoch, model, loss, psnr):
    print("Evaluation")
    run["evaluation/mse"].append(loss)
    run["evaluation/psnr"].append(psnr) 
    run[f"images/reconstructed_{epoch + 1}"].upload(neptune.types.File.as_image(model(val_noisy[0][0])[0].cpu().detach().numpy()))
    print("")
# Log after training
def neptune_val_images(model):
    print("Final evaluation")
    
    for i in range(1, 6):
        run[f"validation/original_{i}"].upload(neptune.types.File.as_image(val_clean[0][i][0].cpu().detach().numpy()))
        run[f"validation/noisy_{i}"].upload(neptune.types.File.as_image(val_noisy[0][i][0].cpu().detach().numpy()))
        run[f"validation/reconstructed_{i}"].upload(neptune.types.File.as_image(model(val_noisy[0][i])[0].cpu().detach().numpy()))

# Limited data
if split == 'limited':
    train_loader = split_data(train_loader, n)[0][0].cpu().numpy()
    test_loader = split_data(test_loader, n)[0][0].cpu().numpy()
    m = 1

class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1),
            torch.nn.Flatten(0, -1),
            torch.nn.Linear(64 * 14 * 14, 128)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(128, 64 * 14 * 14),
            torch.nn.Unflatten(0, (64, 14, 14)),
            torch.nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def run_model(m, epochs):
    # Make overall model
    autoencoder = AE()
    autoencoder.to(device)
    if path != "Test":
        try:
            autoencoder.load_state_dict(torch.load(path))
        except:
            pass

    # neptune_log(-1, autoencoder)
    run["images/original"].upload(neptune.types.File.as_image(val_clean[0][0][0].cpu().numpy()))
    run["images/noisy"].upload(neptune.types.File.as_image(val_noisy[0][0][0].cpu().numpy()))

    # Make n models
    models = [AE() for _ in range(m)]
    

    # Split data
    train, train_noisy = split_data(train_loader, m)
    test, test_noisy = split_data(test_loader, m)

    # Get central weights
    primary_weights = autoencoder.state_dict()
    for model in models:
        model.to(device)
        model.load_state_dict(primary_weights)

    # For the stupid overfitting check
    min_loss = 1
    overfitter = 0
    
    # Train the networks
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")

        for i, model in enumerate(models):
            print(f"Autoencoder {i + 1}")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

            total_loss = 0
            total_psnr = 0

            for j in range(len(train[i])):
                optimizer.zero_grad()
                loss = loss_fn(model(train_noisy[i][j]), train[i][j])
                loss.backward()
                optimizer.step()
            
            for noisy, clean in zip(test_noisy[i], test[i]):
                loss = loss_fn(model(noisy), clean)
                total_loss += loss.item()
                total_psnr += PSNR(model(noisy), clean).item()

            total_loss /= len(test[i])
            total_psnr /= len(test[i])
            run[f"evaluation/autoencoder_{i + 1}/mse"].append(total_loss)
            run[f"evaluation/autoencoder_{i + 1}/psnr"].append(total_psnr)
            print(f"Loss {total_loss}")
            print(f"PSNR {total_psnr}")

        # Aggregate the models
        for key in primary_weights.keys():
            primary_weights[key] = sum([model.state_dict()[key] for model in models]) / m

        for model in models:
            model.load_state_dict(primary_weights)

        autoencoder.load_state_dict(primary_weights)

        print(f"Epoch {epoch + 1} completed")
        
        total_loss = 0
        total_psnr = 0

        for noisy, clean in zip(val_noisy[0], val_clean[0]):
            loss = loss_fn(autoencoder(noisy), clean)
            total_loss += loss.item()
            total_psnr += PSNR(autoencoder(noisy), clean).item()
        
        total_loss /= len(val_noisy[0])
        total_psnr /= len(val_noisy[0])

        print(f"Validation Loss {total_loss}")
        print(f"Validation PSNR {total_psnr}")
        print("")

        neptune_log(epoch, autoencoder, total_loss, total_psnr)

        if epoch % 10 == 0 and epoch != 0 and path != "Test":
            torch.save(autoencoder.state_dict(), f"./Models/model_{n}_{split}_{epoch + 1}.pth")

        # Stupid check for overfitting
        if total_loss < min_loss:
            min_loss = total_loss
            overfitter = 0
        else:
            overfitter += 1
            if overfitter > 10:
                break
    
    neptune_val_images(autoencoder)

    if path != "Test":
        try:
            torch.save(autoencoder.state_dict(), path)
        except:
            torch.save(autoencoder.state_dict(), f"./Models/model_{n}_{split}.pth")

    return autoencoder

t0 = time.time()
autoendocer = run_model(m, epochs)
t1 = time.time()
print("")
print(f"Time: {t1 - t0}")
print("")
run["time"].append(t1 - t0)
run.stop()