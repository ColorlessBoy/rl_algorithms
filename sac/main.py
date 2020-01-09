import gym
import torch
from collections import namedtuple

from models import PolicyNetwork, ValueNetwork, QNetwork
from utils import EnvSampler, hard_update
from sac import SAC

from time import time

# The properties of args:
# 1. env_name (default: HalfCheetah-v2)
# 2. device (default: cpu)
# 3. hidden_sizes (default: (64, 64))
# 4. batch_size (default：256)
# 5. episodes (default: 1000)
# 6. episode_length (default: 1000)
# 7. start_steps (default: 10000)
# 8. seed (default: 0)

def run(args):
    env = gym.make(args.env_name)

    device = torch.device(args.device)

    # 1. Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env.seed(args.seed)

    # 2. Create nets. 
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    v_net = ValueNetwork(state_size, args.hidden_sizes)
    q_net = QNetwork(state_size, action_size, args.hidden_sizes)
    q2_net = QNetwork(state_size, action_size, args.hidden_sizes)
    pi_net = PolicyNetwork(state_size, action_size, args.hidden_sizes)
    vt_net = ValueNetwork(state_size, args.hidden_sizes)
    hard_update(vt_net, v_net)

    env_sampler = EnvSampler(env, args.episode_length)
    alg = SAC(v_net, q_net, q2_net, pi_net, vt_net,
                gamma=0.99, alpha=0.2,
                v_lr=3e-4, q_lr=3e-4, pi_lr=3e-4, vt_lr = 0.02,
                device=device)

    # 3. Warmup.
    start_time = time()
    env_sampler.addSamples(args.start_steps)
    print("Warmup uses {}s.".format(time() - start_time))

    # 4. Start training.
    def get_action(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = pi_net.select_action_detach(state)
        return action.detach().cpu().numpy()[0]

    for episode in range(1, args.episodes+1):
        env_sampler.env_init()
        losses = 0.0
        for _ in range(args.episode_length):
            env_sampler.addSample(get_action)
            batch = env_sampler.sample(args.batch_size)
            losses = alg.update(*batch)
        yield (episode, env_sampler.episode_reward, *losses)


# The properties of args:
# 1. env_name (default: HalfCheetah-v2)
# 2. device (default: cpu)
# 3. hidden_sizes (default: (64, 64))
# 4. batch_size (default：256)
# 5. episodes (default: 1000)
# 6. episode_length (default: 1000)
# 7. start_steps (default: 10000)
# 8. seed (default: 0)

Args = namedtuple( 'Args',
    ('env_name',
    'device',
    'hidden_sizes',
    'batch_size',
    'episodes',
    'episode_length',
    'start_steps',
    'seed')
)

if __name__ == '__main__':
    args = Args(
        'HalfCheetah-v2', # env_name
        'cuda',           # device
        (256,),           # hidden_size
        256,              # batch_size
        1000,             # episodes
        1000,             # episode_length
        10000,            # start_steps
        0,                # seed
    )

    start_time = time()
    for step, reward, q_loss, pi_loss, v_loss, vt_loss in run(args):
        print("Step {}: Reward = {:>10.6f}, q_loss = {:>8.6f}, pi_loss = {:>8.6f}, v_loss = {:>8.6f}, vt_loss = {:>8.6f}".format(
            step, reward, q_loss, pi_loss, v_loss, vt_loss
        ))
    print("Total time: {}s.".format(time() - start_time))