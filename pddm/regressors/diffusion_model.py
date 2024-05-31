import torch
import torch.nn as nn
import numpy as np
import math

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)

class TimeEncodings(nn.Module):
    def __init__(self, dim):
        super(TimeEncodings, self).__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # print('X shape', x.shape)
        # print('emb shape', emb.shape)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP(nn.Module):
    def __init__(self, state_size, action_size, time_embedding_size, output_size, num_fc_layers, depth_fc_layers):
        super(MLP, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.time_embedding_size = time_embedding_size
        self.outputSize = state_size
        self.num_fc_layers = num_fc_layers
        self.depth_fc_layers = depth_fc_layers

        inputSize = 2 * state_size + action_size + time_embedding_size
        self.intermediate_size = depth_fc_layers
        self.model = [nn.Flatten()]
        self.time_mlp = nn.Sequential(
            TimeEncodings(time_embedding_size),
            nn.Linear(self.time_embedding_size, 4*self.time_embedding_size),
            nn.Mish(),
            nn.Linear(4*self.time_embedding_size, self.time_embedding_size)
        )
        self.model.append(nn.Flatten())
        for i in range(num_fc_layers):
            if i == 0:
                self.model.append(nn.Linear(inputSize, self.intermediate_size))
                self.model.append(nn.ReLU())
            else:
                self.model.append(nn.Linear(self.intermediate_size, self.intermediate_size))
                self.model.append(nn.ReLU())
        self.model.append(nn.Linear(self.intermediate_size, state_size))

        self.model =  nn.Sequential(*self.model)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                m.bias.data.fill_(0.0)

        self.model.apply(init_weights)
        self.time_mlp.apply(init_weights)

    def reinitialize(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
                m.bias.data.fill_(0.0)

        self.model.apply(init_weights)
        self.time_mlp.apply(init_weights)
    
    def forward(self, states, time, prev_states, actions):
        # print(inputs.shape)
        time_embed = self.time_mlp(time)
        # first, second = torch.split(inputs, [self.state_size, self.acSize], 2)
        actions = torch.clip(actions, -1, 1)
        # print('states shape: ', states.shape)
        # print('Time shape: ', time.shape)
        # print('prev state shape: ', prev_states.shape)
        # print('actions shape: ', actions.shape)
        inputs = torch.cat((states, time_embed, prev_states, actions), 1)
        # print('Final inputs shape: ', inputs.shape)

        return self.model(inputs)

class GaussianDiffusionModel(nn.Module):
    def __init__(self, state_dim, act_dim, time_dim, out_dim, num_fc_layers, depth_fc_layers, n_timesteps=100):
        super(GaussianDiffusionModel, self).__init__()
        self.model = MLP(state_size=state_dim, action_size=act_dim, time_embedding_size=time_dim, output_size=out_dim, num_fc_layers=num_fc_layers, depth_fc_layers=depth_fc_layers)
        
        self.state_dim = state_dim
        self.out_dim = out_dim

        self.n_timesteps = n_timesteps

        self.betas = cosine_beta_schedule(n_timesteps).cuda()
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).cuda(), self.alphas_cumprod[:-1]])

        # # calculating x_t from x_0
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1malphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

        # # posterior variance
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.recip_sqrt_alphas = 1. / torch.sqrt(self.alphas)


    def reinitialize(self):
        self.model.reinitialize()

    def sample_mean_variance(self, x_t, t, prev_states, actions):
        noise_from_model = self.model(x_t, t, prev_states, actions)

        recip_sqrt_alpha_t = extract(self.recip_sqrt_alphas, t, x_t.shape)
        beta_t = extract(self.betas, t, x_t.shape)
        sqrt_1malphas_cumprod_t = extract(self.sqrt_1malphas_cumprod, t, x_t.shape)

        mean = recip_sqrt_alpha_t * (x_t - (beta_t * noise_from_model) / sqrt_1malphas_cumprod_t)
        variance = extract(self.posterior_variance, t, x_t.shape)

        return mean, variance

    @torch.no_grad()
    def sample_step(self, x_t, t, prev_states, actions):
        noise = torch.randn_like(x_t)

        mean, variance = self.sample_mean_variance(x_t, t, prev_states, actions)
        nonzero_mask = (1 - (t == 0).float()).reshape(x_t.shape[0], *((1,) * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.sqrt(variance) * noise

    @torch.no_grad()
    def sample(self, prev_states, actions):
        # print('Device: ', prev_states.device)
        # print(prev_states.shape)
        # print(actions.shape)
        batch_size = actions.shape[0]
        x = torch.rand((batch_size, self.state_dim)).to(prev_states.device)
        # print(x.shape)
        for i in reversed(range(0, self.n_timesteps)):
            time_steps = np.ones((batch_size,)) * i
            # print(torch.tensor().shape)
            x = self.sample_step(x, torch.tensor(time_steps).long().cuda(), prev_states, actions)

        return x

    #------------------------------------------ training ------------------------------------------#
    def get_x_t(self, x_start, t):
        noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_1malphas_cumprod_t = extract(self.sqrt_1malphas_cumprod, t, x_start.shape)

        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_1malphas_cumprod_t * noise
        return x_t


    def loss(self, x_start, prev_states, actions):
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,)).to(x_start.device).long()
        # print('Time shape', t.shape) 

        noise = torch.randn_like(x_start)
        
        x_t = self.get_x_t(x_start, t)

        noise_from_model = self.model(x_t, t, prev_states, actions)

        return torch.nn.functional.mse_loss(noise_from_model, noise)

class DiffusionEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, time_dim, num_fc_layers, depth_fc_layers, ensemble_size=1):
        super(DiffusionEnsemble, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.time_dim = time_dim
        self.ensemble_size = ensemble_size
        self.num_fc_layers = num_fc_layers
        self.depth_fc_layers = depth_fc_layers
        # self.reg = torch.tensor(reg.T).float().cuda()

        self.intermediate_size = depth_fc_layers
        # self.criterion = nn.MSELoss()

        self.members = []
        for i in range(self.ensemble_size):
            self.members.append(GaussianDiffusionModel(state_dim, action_dim, time_dim, state_dim, num_fc_layers, depth_fc_layers, n_timesteps=100).cuda())

        self.members = nn.ModuleList(self.members)

    def reinitialize(self):
        for member in self.members:
            member.reinitialize()

    def forward_train(self, inputs, labels): #[bs, K, sa]
        inputs = torch.squeeze(inputs, 1)
        # print('After squeeze: ', inputs.shape)
        inputs_tiled = torch.tile(inputs, (self.ensemble_size, 1, 1))
        outputs_tiled = torch.tile(labels, (self.ensemble_size, 1, 1))

        prev_states = inputs_tiled[:, :, 0:self.state_dim]
        actions = inputs_tiled[:, :, self.state_dim:]

        loss = 0
        for i in range(self.ensemble_size):
            loss += self.members[i].loss(outputs_tiled[i], prev_states[i], actions[i])
    
        return loss/self.ensemble_size


    def forward_eval(self, inputs, labels): #[bs, K, sa]
        inputs = torch.squeeze(inputs, 1)
        inputs_tiled = torch.tile(inputs, (self.ensemble_size, 1, 1))
        outputs_tiled = torch.tile(labels, (self.ensemble_size, 1, 1))

        prev_states = inputs_tiled[:, :, 0:self.state_dim]
        actions = inputs_tiled[:, :, self.state_dim:]

        with torch.no_grad():
            loss = 0
            for i in range(self.ensemble_size):
                loss += self.members[i].loss(outputs_tiled[i], prev_states[i], actions[i])
        
        return loss/self.ensemble_size

    def forward_sim(self, inputs): #[ens, bs, K, sa]
        inputs = torch.squeeze(inputs, 2)
        # print('Forward sim: ', inp)
        prev_states = inputs[:, :, 0:self.state_dim]
        actions = inputs[:, :, self.state_dim:]
        with torch.no_grad():
            outputs = []
            for i in range(self.ensemble_size):
                model = self.members[i].eval()
                output_i = model.sample(prev_states[i], actions[i])
                outputs.append(output_i.cpu().detach().numpy())
    
        return np.array(outputs)