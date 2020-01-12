import gym
import torch
from collections import namedtuple
import os
import csv
from time import time
import random


from models import PolicyNetwork, ValueNetwork, QNetwork
from utils import EnvSampler, hard_update
from sac import SAC

# The properties of args:
# 1. env_name (default: HalfCheetah-v2)
# 2. device (default: cpu)
# 3. hidden_sizes (default: (64, 64))
# 4. batch_size (default：256)
# 5. total_steps (default: 1000000)
# 6. max_episode_length (default: 1000)
# 7. start_steps (default: 10000)
# 8. seed (default: 0)

def run(args):
    env = gym.make(args.env_name)

    device = torch.device(args.device)

    # 1. Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)

    # 2. Create nets. 
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    v_net = ValueNetwork(state_size, args.hidden_sizes, 
        activation=args.activation).to(device)
    q_net = QNetwork(state_size, action_size, args.hidden_sizes,
        activation=args.activation).to(device)
    q2_net = QNetwork(state_size, action_size, args.hidden_sizes,
        activation=args.activation).to(device)
    pi_net = PolicyNetwork(state_size, action_size, args.hidden_sizes,
        activation=args.activation, output_activation=args.output_activation).to(device)
    vt_net = ValueNetwork(state_size, args.hidden_sizes,
        activation=args.activation).to(device)
    hard_update(vt_net, v_net)

    env_sampler = EnvSampler(env, args.max_episode_length)
    alg = SAC(v_net, q_net, q2_net, pi_net, vt_net,
                gamma=0.99, alpha=0.2,
                v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, vt_lr = args.vt_lr,
                device=device)

    # 3. Warmup.
    start_time = time()
    env_sampler.addSamples(args.start_steps)
    print("Warmup uses {}s.".format(time() - start_time))

    # 4. Start training.
    def get_action(state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _ = pi_net.select_action(state)
        return action.cpu().numpy()[0]

    def get_mean_action(state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = pi_net.get_mean_action(state)
        return action.cpu().numpy()[0]

    def save_models(model, model_name, step):
        folder_path = './model/{}'.format(model_name)
        file_name = 'step{}_time{}.pth.tar'.format(step, time())
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        full_name = os.path.join(folder_path, file_name)
        torch.save(model.state_dict(), full_name)

    trajectory = 0
    model_save_file = 'models'
    for step in range(1, args.total_steps+1):
        done, episode_reward = env_sampler.addSample(get_action)
        batch = env_sampler.sample(args.batch_size)
        losses = alg.update(*batch)
        if done:
            trajectory += 1
            test_reward = None
            if trajectory % args.test_frequency == 0:
                start_time = time()
                save_models(pi_net, 'pi_net', step)
                test_reward = env_sampler.test(get_mean_action, 10)
                print('Testing policy uses {}s.'.format(time() - start_time))
            yield (step, episode_reward, *losses, test_reward)


# The properties of args:
# 1. env_name (default: HalfCheetah-v2)
# 2. device (default: cpu)
# 3. hidden_sizes (default: (64, 64))
# 4. batch_size (default：256)
# 5. total_steps (default: 1000000)
# 6. max_episode_length (default: 1000)
# 7. start_steps (default: 10000)
# 8. seed (default: 0)
# 9. test_frequency (default: 10)
# 10. activation (default: torch.relu)
# 11. output_activation (default: torch.tanh)

Args = namedtuple( 'Args',
    ('env_name',
    'device',
    'hidden_sizes',
    'batch_size',
    'total_steps',
    'max_episode_length',
    'start_steps',
    'seed',
    'vt_lr',
    'test_frequency',
    'activation',
    'output_activation')
)

if __name__ == '__main__':
    alg_args = Args(
        'HalfCheetah-v2', # env_name
        'cuda',           # device
        (400, 300),       # hidden_size
        100,              # batch_size
        1000000,          # total_steps
        1000,             # max_episode_length
        10000,            # start_steps
        0,                # seed
        0.005,            # vt_lr
        10,               # test_frequency
        torch.relu,       # activation
        torch.tanh,       # output_activation
    )

    logdir = "./logs/algo_sac/env_{}".format(alg_args.env_name)
    file_name = 'algo_sac_env_{}_batch{}_seed{}_time{}.csv'.format(alg_args.env_name, alg_args.batch_size, alg_args.seed, time())
    test_file_name = 'test_algo_sac_env_{}_batch{}_seed{}_time{}.csv'.format(alg_args.env_name, alg_args.batch_size, alg_args.seed, time())
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    full_name = os.path.join(logdir, file_name)
    test_full_name = os.path.join(logdir, test_file_name)

    csvfile = open(full_name, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['step', 'reward'])

    test_csvfile = open(test_full_name, 'w')
    test_writer = csv.writer(test_csvfile)
    writer.writerow(['step', 'test_reward'])

    start_time = time()
    for step, reward, q_loss, pi_loss, v_loss, vt_loss, test_reward in run(alg_args):
        writer.writerow([step, reward])
        print("Step {}: Reward = {:>10.6f}, q_loss = {:>8.6f}, pi_loss = {:>8.6f}, v_loss = {:>8.6f}, vt_loss = {:>8.6f}".format(
            step, reward, q_loss, pi_loss, v_loss, vt_loss
        ))
        if test_reward is not None:
            test_writer.writerow([step, test_reward])
            print("Step {}: Test reward = {}".format(step, test_reward))
    print("Total time: {}s.".format(time() - start_time))