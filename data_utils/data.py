import torch
import matplotlib.pyplot as plt
import IPython.display as ipd
from IPython.display import Audio
import os
import librosa
import torch.nn.functional as F
import numpy as np
import plotly.graph_objs as go
from tqdm import tqdm
import sys
from scipy.spatial.distance import cosine
sys.path.append('/home/yuriih/LLTM/training_utils/loss_functions')
from training_utils.loss_functions import calculate_timbre_difference

def check_dataset(sample_indx, effect_indx, vocoder, standard_scalar, valid_loader):
    # Check if the data processed correctly
    x = valid_loader.dataset[0]
    x_np = x[0][1].cpu().numpy().reshape(1,409*64)
    x_unscaled = standard_scalar.inverse_transform(x_np)
    x_unscaled = x_unscaled.reshape(409, 64)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(x_unscaled.T, aspect='auto', origin='lower')
    plt.title('Original Spectrogram')

    with torch.no_grad():
        x_audio = vocoder(torch.tensor(x_unscaled)).cpu().numpy()


    ipd.display(Audio(x_audio, rate=16000))
    
    
def reconstruction_vis(i, audio_dir, vocoder, device, valid_loader, standard_scalar, time_bins_num, freq_bins_num, vae):
    audio_files = os.listdir(audio_dir)
    assert i < len(audio_files), f"i must be less than {len(audio_files)}"
    
    sr = 16000
    original_audio, _= librosa.load(os.path.join(audio_dir, audio_files[i]), sr=sr)

    # Get the clean version of the audio (no effects)
    x = valid_loader.dataset[i][0][0].unsqueeze(0).to(device)

    vae.eval()
    with torch.no_grad():
        x_recon = vae(x)[0].cpu()

    x = x.cpu()

    loss_valid = F.mse_loss(x, x_recon, reduction='mean')

    x = standard_scalar.inverse_transform(x.numpy()).reshape(time_bins_num, freq_bins_num )
    x_recon = standard_scalar.inverse_transform(x_recon.numpy()).reshape(time_bins_num, freq_bins_num)


    print('MSE:', loss_valid.item())

    # Plot the original spectrogram
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(x.T, aspect='auto', origin='lower')
    plt.title('Original Spectrogram')
    plt.colorbar(im1, ax=plt.gca())

    # Plot the reconstructed spectrogram
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(x_recon.T, aspect='auto', origin='lower')
    plt.title('Reconstructed Spectrogram')
    plt.colorbar(im2, ax=plt.gca())

    plt.tight_layout()
    plt.show()



    with torch.no_grad():
        x_audio = vocoder(torch.tensor(x)).cpu().numpy()
        x_recon_audio = vocoder(torch.tensor(x_recon)).cpu().numpy()


    print('Audio before Hifi-GAN:')
    ipd.display(Audio(original_audio[:3*sr], rate=sr))

    print('Audio after Hifi-GAN:')
    ipd.display(Audio(x_audio[:3*sr], rate=sr))

    print('Audio after VAE and Hifi-GAN (reconstruction):')
    ipd.display(Audio(x_recon_audio[:3*sr], rate=sr))
    
    
    
def latent_space_3D_vis(reducer, vae, valid_loader, device):
    labels = []
    z_list = []
    vae.eval()
    with torch.no_grad():
        for batch_data in tqdm(valid_loader):
            batch_data_gpu = torch.concatenate([batch_data[0].select(1, 0), 
                                            batch_data[0].select(1, 1),
                                            batch_data[0].select(1, 2)]).to(device)
            n = batch_data[0].shape[0]
            batch_labels_gpu = torch.concatenate([torch.tensor([[1,0,0]]*n),
                                                  torch.tensor([[0,1,0]]*n),
                                                  torch.tensor([[0,0,1]]*n)]).to(device)
            mu, std  = vae.encode(batch_data_gpu)
            z_list.append(mu.cpu().numpy())
            labels.append(batch_labels_gpu.cpu().numpy())

    z_array = np.concatenate(z_list, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels = np.argmax(labels, axis=1)
    labels = np.select([labels==0, labels==1, labels==2], ['no effect', 'highpass', 'lowpass'], default='unknown')
    test_names = [ name[16:-4] for name in np.array(os.listdir('datasets/nsynth-subset/nsynth-test-audios/audio'))]
    test_names = test_names[:z_array.shape[0]//3]
    test_names = np.repeat(test_names, 3)


    z_low_dim = reducer.fit_transform(z_array)
    

    unique_labels = np.unique(labels)
    label_color_map = {
        'no effect': 'blue',
        'highpass': 'red',
        'lowpass': 'green',
    }

    data_traces = []

    for ulabel in unique_labels:
        idx = np.where(labels == ulabel)[0] 
        trace = go.Scatter3d(
            x=z_low_dim[idx, 0],
            y=z_low_dim[idx, 1],
            z=z_low_dim[idx, 2],
            mode='markers',
            name=str(ulabel),
            marker=dict(
                size=5,
                color=label_color_map[ulabel],
                opacity=0.9
            ),
            hovertemplate=(
                "Name: %{text}<br>" +
                "Label: " + str(ulabel) + "<br>" +
                "x: %{x}<br>" +
                "y: %{y}<br>" +
                "z: %{z}"
            ),
            text=[test_names[i] for i in idx]  
        )
        data_traces.append(trace)  

    layout = go.Layout(
        margin=dict(l=10, r=10, b=0, t=50),
        hovermode='closest',
        scene=dict(  
            xaxis=dict(title='x'),
            yaxis=dict(title='y'),
            zaxis=dict(title='z')
        )
    )


    fig = go.Figure(data=data_traces, layout=layout)

    fig.show()

def latent_space_2D_vis(reducer, vae, valid_loader, device):
    labels = []
    z_list = []
    
    vae.eval()
    with torch.no_grad():
        for batch_data in tqdm(valid_loader):
            batch_data_gpu = torch.concatenate([batch_data[0].select(1, 0), 
                                            batch_data[0].select(1, 1),
                                            batch_data[0].select(1, 2)]).to(device)
            n = batch_data[0].shape[0]
            batch_labels_gpu = torch.concatenate([torch.tensor([[1,0,0]]*n),
                                                  torch.tensor([[0,1,0]]*n),
                                                  torch.tensor([[0,0,1]]*n)]).to(device)
            mu, std = vae.encode(batch_data_gpu)
            z_list.append(mu.cpu().numpy())
            labels.append(batch_labels_gpu.cpu().numpy())

    z_array = np.concatenate(z_list, axis=0)
    labels = np.concatenate(labels, axis=0)
    labels = np.argmax(labels, axis=1)
    labels = np.select([labels == 0, labels == 1, labels == 2], ['no effect', 'highpass', 'lowpass'], default='unknown')

    test_names = [ name[16:-4] for name in np.array(os.listdir('datasets/nsynth-subset/nsynth-test-audios/audio'))]
    test_names = test_names[:z_array.shape[0] // 3]
    test_names = np.repeat(test_names, 3)

    z_low_dim = reducer.fit_transform(z_array)

    unique_labels = np.unique(labels)
    label_color_map = {
        'no effect': 'blue',
        'highpass': 'red',
        'lowpass': 'green',
    }
    unique_names = np.unique(test_names)
    name_indices = {name: np.flatnonzero(test_names == name) for name in unique_names}

    data_traces = []
    for name, indices in name_indices.items():
        x_coords = z_low_dim[indices, 0]
        y_coords = z_low_dim[indices, 1]
        current_labels = labels[indices]

    for ulabel in unique_labels:
        idx = np.where(labels == ulabel)[0]
        marker_trace = go.Scatter(
            x=z_low_dim[idx, 0],
            y=z_low_dim[idx, 1],
            mode='markers',
            name=str(ulabel),
            marker=dict(
                size=5,
                color=label_color_map[ulabel], 
                opacity=0.9
            ),
            text=[test_names[i] for i in idx],
            hovertemplate=(
                "Name: %{text}<br>" +
                "Label: " + str(ulabel) + "<br>" +
                "x: %{x}<br>" +
                "y: %{y}<br>"
            )
        )
        data_traces.append(marker_trace)

    layout = go.Layout(
        margin=dict(l=10, r=10, b=0, t=50),
        hovermode='closest',
        xaxis=dict(title='x'),
        yaxis=dict(title='y')
    )

    fig = go.Figure(data=data_traces, layout=layout)
    
    fig.show()
    
    
def calculate_timbre_effect_masks(vae, train_loader, device, latent_dim, batch_size=64):
    effect_difference_mask = torch.zeros(latent_dim).to(device)
    timbre_difference_mask = torch.zeros(latent_dim).to(device)
    total_samples = 0
    vae.eval()
    with torch.no_grad():
        for batch_data in train_loader:
            
            batch_data_gpu_all = torch.cat([batch_data[0].select(1, i) for i in range(3)], dim=0).to(device)
            recon_batch_all, mu_all, log_var_all = vae(batch_data_gpu_all)
            mu_el, mu_effect1, mu_effect2= mu_all.split(batch_data[0].size(0))

            # Effect loss
            effect_difference = (torch.abs(mu_effect1 - mu_el).mean(dim=0) \
                                + torch.abs(mu_effect2 - mu_el).mean(dim=0)
                                + torch.abs(mu_effect1 - mu_effect2).mean(dim=0)) / 3
            effect_difference_norm = effect_difference / torch.norm(effect_difference)
        
            effect_difference_mask += effect_difference_norm
            
            # Timbre loss
            timbre_difference = (calculate_timbre_difference(mu_el, device, batch_size) \
                                + calculate_timbre_difference(mu_effect1, device, batch_size) \
                                + calculate_timbre_difference(mu_effect2, device, batch_size) ) / 3
            timbre_difference_norm = timbre_difference / torch.norm(timbre_difference)
            timbre_difference_mask+= timbre_difference_norm
    



    num_batches = len(train_loader) - 1
    effect_difference_mask /= num_batches
    timbre_difference_mask /= num_batches


    # Normalize the masks
    effect_difference_mask_norm = effect_difference_mask.detach().cpu().numpy()
    timbre_difference_mask_norm = timbre_difference_mask.detach().cpu().numpy()
    return effect_difference_mask_norm, timbre_difference_mask_norm




def convert_to_audio(temp_spec, description, device, standard_scalar, vocoder, time_bins_num, freq_bins_num, vae):
    temp_spec = temp_spec.unsqueeze(0).to(device)

    vae.eval()
    with torch.no_grad():
        recon, mu, log_var = vae(temp_spec)
    
    temp_spec = standard_scalar.inverse_transform(recon.cpu().numpy()).reshape(time_bins_num, freq_bins_num)

    im3 = plt.imshow(temp_spec.T, aspect='auto', origin='lower')
    plt.title('Effect Spectrogram')
    plt.colorbar(im3, ax=plt.gca())
    with torch.no_grad():
        temp_audio = vocoder(torch.tensor(temp_spec)).cpu().numpy()

    sr = 16000

    print(description)
    ipd.display(Audio(temp_audio, rate=sr))


def copy_effects_vis(input_spec, target_spec, input_approx_spec, clean_target, target_cutoff, effect_mask_binary, vae, device, standard_scalar, vocoder, time_bins_num, freq_bins_num, reducer_pca):
    
    data = torch.stack([input_spec, target_spec, input_approx_spec, clean_target], dim=0).to(device)
    vae.eval()
    with torch.no_grad():
        recon, mu, log_var = vae(data)

    input_embeddings, target_embeddings, input_approx_embeddings, clean_target_embeddings = mu.split(1)
    input_recon, target_recon, input_approx_recon, clean_target_recon = recon.split(1)

    input_embeddings_modified = input_embeddings.clone()

    #effect_mask_binary_tensor = torch.tensor(effect_mask_binary).to(device)
    #movement_vector = target_embeddings[0] * effect_mask_binary_tensor
    #input_embeddings_modified[:, effect_mask_binary > threshold] = movement_vector[effect_mask_binary > threshold]
    #Change clean embeddings based on the effect mask
    input_embeddings_modified[:, effect_mask_binary == 1] = target_embeddings[:, effect_mask_binary == 1]
    
    
    
    # 2d Latent space
    input_embeddings_np = input_embeddings.cpu().numpy()
    target_embeddings_np = target_embeddings.cpu().numpy()
    input_approx_embeddings_np = input_approx_embeddings.cpu().numpy()
    input_embeddings_modified_np = input_embeddings_modified.cpu().numpy()
    clean_target_embeddings_np = clean_target_embeddings.cpu().numpy()

    input_embeddings_2d = reducer_pca.transform(input_embeddings_np)
    target_embeddings_2d = reducer_pca.transform(target_embeddings_np)
    input_approx_embeddings_2d = reducer_pca.transform(input_approx_embeddings_np)
    input_modified_embeddings_2d = reducer_pca.transform(input_embeddings_modified_np)
    clean_target_embeddings_2d = reducer_pca.transform(clean_target_embeddings_np)

    input_target_cos_distance = cosine(input_embeddings_modified_np[0], target_embeddings_np[0])
    input_approx_cos_distance = cosine(input_embeddings_modified_np[0], input_approx_embeddings_np[0])

    # Calculate Euclidean distances
    input_target_eucl_distance = np.linalg.norm(input_embeddings_modified_np[0] - target_embeddings_np[0])
    input_approx_eucl_distance = np.linalg.norm(input_embeddings_modified_np[0] - input_approx_embeddings_np[0])


    print(f'Result-Target: \n Cos: {input_target_cos_distance:.2f}, Eucl: {input_target_eucl_distance:.2f}')
    print(f'Result-Approx: \n Cos: {input_approx_cos_distance:.2f}, Eucl: {input_approx_eucl_distance:.2f}')
    
    # Plot the embeddings
    plt.figure(figsize=(8, 6))
    plt.scatter(input_embeddings_2d[:, 0], input_embeddings_2d[:, 1], color='blue', label='Input', alpha=0.5, s=100)
    plt.scatter(target_embeddings_2d[:, 0], target_embeddings_2d[:, 1], color='red', label='Target', alpha=0.5, s=200, marker='x')
    plt.scatter(input_approx_embeddings_2d[:, 0], input_approx_embeddings_2d[:, 1], color='red', label='Groundtruth Effect Input', alpha=0.5, s=150)
    plt.scatter(input_modified_embeddings_2d[:, 0], input_modified_embeddings_2d[:, 1], color='red', label='Modified Input', alpha=0.8, marker='*',s=300)
    plt.scatter(clean_target_embeddings_2d[:, 0], clean_target_embeddings_2d[:, 1], color='blue', label='No Effect Target', alpha=0.5, marker='x',s=150)

    # Draw lines between the points
    plt.plot([input_modified_embeddings_2d[:, 0], target_embeddings_2d[:, 0]], [input_modified_embeddings_2d[:, 1], target_embeddings_2d[:, 1]], 'k-', alpha=0.4)
    plt.plot([input_modified_embeddings_2d[:, 0], input_approx_embeddings_2d[:, 0]], [input_modified_embeddings_2d[:, 1], input_approx_embeddings_2d[:, 1]], 'k-', alpha=0.4)


    plt.legend()

    plt.legend()
    plt.show()
    
    

    
    # Spectograms and Audios
    vae.eval()
    with torch.no_grad():
        decoded_input_modified = vae.decoder(input_embeddings_modified)


    input_modified_recon_unscaled = standard_scalar.inverse_transform(decoded_input_modified.cpu().numpy()).reshape(time_bins_num, freq_bins_num)
    input_recon_unscaled = standard_scalar.inverse_transform(input_recon.cpu().numpy()).reshape(time_bins_num, freq_bins_num)
    input_approx_recon_unscaled = standard_scalar.inverse_transform(input_approx_recon.cpu().numpy()).reshape(time_bins_num, freq_bins_num)
    target_recon_unscaled = standard_scalar.inverse_transform(target_recon.cpu().numpy()).reshape(time_bins_num, freq_bins_num)
    clean_target_recon_unscaled = standard_scalar.inverse_transform(clean_target_recon.cpu().numpy()).reshape(time_bins_num, freq_bins_num)
    
    
    mse = ((input_modified_recon_unscaled - input_approx_recon_unscaled)**2).mean()
    print(f'MSE between Result and Groundtruth: {mse:.6f}')

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    im1 = plt.imshow(input_recon_unscaled.T, aspect='auto', origin='lower')
    plt.title('Input')
    plt.colorbar(im1, ax=plt.gca())

    plt.subplot(2, 2, 3)
    im2 = plt.imshow(input_modified_recon_unscaled.T, aspect='auto', origin='lower')
    plt.title('Modified Input')
    plt.colorbar(im2, ax=plt.gca())
    
    plt.subplot(2, 2, 4)
    im2 = plt.imshow(input_approx_recon_unscaled.T, aspect='auto', origin='lower')
    plt.title(f'Groundtruth Effect Input')
 
    plt.colorbar(im2, ax=plt.gca())

    plt.subplot(2, 2, 2)
    im3 = plt.imshow(target_recon_unscaled.T, aspect='auto', origin='lower')
    plt.title(f'Target (cutoff: {target_cutoff} Hz)')

    plt.colorbar(im3, ax=plt.gca())

    plt.tight_layout()
    plt.show()

    with torch.no_grad():
        clean_audio = vocoder(torch.tensor(input_recon_unscaled)).cpu().numpy()
        effect_audio = vocoder(torch.tensor(target_recon_unscaled)).cpu().numpy()
        clean_effect_audio = vocoder(torch.tensor(clean_target_recon_unscaled)).cpu().numpy()
        copied_effect_audio = vocoder(torch.tensor(input_modified_recon_unscaled)).cpu().numpy()
        clean_approx_audio = vocoder(torch.tensor(input_approx_recon_unscaled)).cpu().numpy()

    sr = 16000

    print('Input:')
    ipd.display(Audio(clean_audio[:3*sr], rate=sr))

    print('Target')
    ipd.display(Audio(effect_audio[:3*sr], rate=sr))
    
    print('No Effect Target')
    ipd.display(Audio(clean_effect_audio[:3*sr], rate=sr))

    print('Modified Input (Result)')
    ipd.display(Audio(copied_effect_audio[:3*sr], rate=sr))
    
    print('Groundtruth Effect Input')
    ipd.display(Audio(clean_approx_audio[:3*sr], rate=sr))
    