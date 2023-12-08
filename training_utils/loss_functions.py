import torch
import torch.nn.functional as F

def KLD_function(mu, std):
    return -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2)

    

def calculate_timbre_difference(mu_el, device, batch_size=64):
    # Timbre loss
    # mu_el_expanded_a has shape (batch_size, 1, 256)
    # mu_el_expanded_b has shape (1, batch_size, 256)
    mu_el_expanded_a = mu_el.unsqueeze(1)
    mu_el_expanded_b = mu_el.unsqueeze(0)

    # Compute the differences using broadcasting
    # diff has shape (batch_size, batch_size, 256)
    diff = torch.abs(mu_el_expanded_a - mu_el_expanded_b)

    mask = torch.triu(torch.ones(batch_size, batch_size), diagonal=1).bool().to(device)
    diff_masked = diff[mask]
    return diff_masked.mean(0)

def disentangle_loss_function(mu_el, mu_effect1, mu_effect2, device, batch_size=64):
    # Effect loss
    effect_difference = (torch.abs(mu_el - mu_effect1).mean(dim=0) \
                        + torch.abs(mu_el - mu_effect2).mean(dim=0)
                        + torch.abs(mu_effect1 - mu_effect2).mean(dim=0)) / 3
            
    # Timbre loss
    timbre_difference = (calculate_timbre_difference(mu_el, device, batch_size) \
                        + calculate_timbre_difference(mu_effect1, device, batch_size) \
                        + calculate_timbre_difference(mu_effect2, device, batch_size)) / 3
    # Normalize to focus on the orientation
    effect_difference_norm = effect_difference / torch.norm(effect_difference)
    timbre_difference_norm = timbre_difference / torch.norm(timbre_difference)
    # Disentangle timbre and effect
    return torch.dot(effect_difference_norm, timbre_difference_norm)


def compute_KL_weight(epoch, num_epochs, initial_beta, final_beta, power):
    return initial_beta + (final_beta - initial_beta) * ((epoch / num_epochs) ** power)