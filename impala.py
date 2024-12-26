import os
import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F
from torch import multiprocessing as mp
import torch.optim as optim
import logging
import time 

os.environ["OMP_NUM_THREADS"] = "1"  # for multiprocessing.

 
def setup_logger(name, log_file, level=logging.INFO):
    handler = logging.FileHandler(log_file, mode='w')  # 각 프로세스별 별도의 로그 파일 생성
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def compute_pg_loss(pi_logits, actions, pg_advantages):
    cross_entropy = F.nll_loss(F.log_softmax(pi_logits, dim=1), target=actions, reduction="none")
    return torch.sum(cross_entropy * pg_advantages)        


def compute_v_trace(mu_logits,  # behaviour policy
                    pi_logits,  # target policy
                    actions, rewards, values, dones, # trajectories
                    gamma, rho_bar = 1.0, c_bar = 1.0, rho_bar_pg = 1.0  # parameter, thresholds
                    ):
    with torch.no_grad():
        discounts = dones * gamma  
        values = values.detach()
        values = values.squeeze(1)
    
        pi_probs = torch.softmax(pi_logits, dim=1)  # dim=1 행, 0 열, -1 마지막 성분
        mu_probs = torch.softmax(mu_logits, dim=1)
        # tensor([[0.5133, 0.4867],
                # [0.5107, 0.4893],
                # [0.5131, 0.4869],
        rhos = (pi_probs / mu_probs).gather(dim=1, index=actions.unsqueeze(1)).squeeze(1)
        rhos_clip = torch.clamp(rhos, max = rho_bar) # IS weights
        cs_clip = torch.clamp(rhos, max = c_bar)

        v_last = values[-1]  # for v_{t+1}, tensor(0.1426) 
        values_2_to_last = torch.cat([values[1:], v_last.unsqueeze(0)])  # v_2, ..., v_t+1, tensor([0.0785, 0.0592, 0.0775,...
        delta_V_s = rhos_clip * (rewards + discounts * values_2_to_last - values)  # tensor([-0.0497, -0.0685, -0.0492, ... ]), (1) a temporal difference for V

        # Remark 1.
        tmp = torch.zeros_like(values[1])  # tensor(0.)  v_{s+n} - V(x_{s+n}), 
        v_s_minus_v_x_s = []
        for t in reversed(range(len(delta_V_s))):
                tmp = delta_V_s[t] + discounts[t] * cs_clip[t] * tmp
                v_s_minus_v_x_s.insert(0, tmp)
        v_s_minus_v_x_s = torch.tensor(v_s_minus_v_x_s)  # tensor([-0.0495, -0.0682, -0.0490, ... ])
        v_s = v_s_minus_v_x_s + values # torch.add(v_s_minus_v_x_s, values), tensor([0.0102, 0.0103, 0.0102, 0.0104])

        # pg_advantages
        rhos_clip_pg = torch.clamp(rhos, max = rho_bar_pg)
        q_s = rewards + discounts * values_2_to_last # q_s = r_s + gamma * v_{s+1}
        pg_advantages = rhos_clip_pg * (q_s - values)  # tensor([-0.0495, -0.0682, -0.0490, ... ])

        return v_s, pg_advantages, rhos


def actor_process(idx, actor_model, learner_model, queue, unroll_length, episode, total_time, lock, terminate_event):
    logger = setup_logger(f'Actor_{idx}', f'actor_{idx}.log')
    logger.info('Actor started')
    
    torch.manual_seed(1234*(idx+1))  # torch seed 고정

    # Environmnet
    env_name = "CartPole-v1"
    env = gym.make(env_name)
    
    state, _ = env.reset()  # initial state
    state = torch.tensor(state, dtype=torch.float32)  # tensor([0.0088, 0.0197, 0.0500, 0.0346])
    done = False  # if True, then episode terminates
    trajectory = []
    episodic_return = 0  # episodic return
    
    while True:

        logits, values = actor_model(state)  # tensor([-0.1016,  0.0095], grad_fn=<ViewBackward0>), tensor([0.1504], grad_fn=<ViewBackward0>)

        action = torch.multinomial(torch.softmax(logits, dim=0), num_samples=1).item()
        next_state, reward, done = env.step(action)[:3]  # use only termination, excluding truncation. So, we can get the rewards over 500 during the episodic.
        next_state = torch.tensor(next_state, dtype=torch.float32)
        trajectory.append((next_state, action, reward, done, logits.detach()))
        episodic_return += reward

        with lock:
            total_time.value += 1
        
        if done:  # if done is True, then the episode is over.
            logger.info(f'episodic returns: {episodic_return}, total_time: {total_time.value}, episode: {episode.value}')
            with lock:
                episode.value += 1

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32)  # tensor([0.0088, 0.0197, 0.0500, 0.0346])
            done = False
            episodic_return = 0
        else:
            state = next_state


        if len(trajectory) >= unroll_length:  # 데이터 수집
            with lock:
                # queue.put((trajectory, actor_model.state_dict()))
                queue.put(trajectory) 
                trajectory = []
                actor_model.load_state_dict(learner_model.state_dict())  # parameter 동기화

        
        # terminate conditon
        if episode.value >= max_episode or terminate_event.is_set(): 
            terminate_event.set()
            break

    logger.info(f'Actor {idx} finished, total_time: {total_time.value}, episode: {episode.value}, terminate condition : {terminate_event.is_set()}')
        

def learner_process(learner_model, queue, episode, batch_size, optimizer, lock, terminate_event):
    logger = setup_logger('Learner', 'learner.log')
    logger.info('Learner started')
    

    trajectories = [] 
    while True:
        while len(trajectories) < batch_size:
            ### learner batching time 측정 시작 ###
            start = time.perf_counter()       
            with lock:
                if not queue.empty():  # True(비어 있음)
                    trajectory = queue.get()  # Queue에서 항목 제거
                    trajectories.extend(trajectory)   # 제거한 항목 추가        
        if len(trajectories) >= batch_size:
            batch = trajectories[:batch_size]
            end = time.perf_counter()
            learner_batching_time = end - start
            ### learner batching time 측정 종료 ###

            trajectories = trajectories[batch_size:]
        
            states, actions, rewards, dones, logits = zip(*batch) 
            # trajectory data from actor
            states = torch.stack(states)     # tensor([[ 0.0347,  0.2377, -0.0142, -0.2862],
                                             #         [ 0.0395,  0.0428, -0.0199,  0.0019],
            actions = torch.tensor(actions)  # 정수형으로 tensor([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
            rewards = torch.tensor(rewards, dtype=torch.float32) # tensor([1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
            dones = torch.tensor(dones, dtype=torch.float32)  # tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
            logits = torch.stack(logits)     # tensor([[ 0.0390, -0.0055], # behaivour logits
                                             #         [ 0.0398, -0.0134],
            
            ### learner forward time 측정 시작 ###
            start = time.perf_counter()        
            pi_logits, values = learner_model(states)
            # values:
            # tensor([[0.0597],
            #         [0.0785],
            #         [0.0592],
            #         [0.0775]], grad_fn)
            end = time.perf_counter()
            learner_forward_time = end - start       
            ### learner forward time 측정 종료 ###
 
 
            # target logits
            pi_logits, values = learner_model(states)
            # vtrace
            v_s, pg_advantages, rhos = compute_v_trace(logits, pi_logits, actions, rewards, values, dones, gamma, clipping_rho, clipping_c, clipping_rho_bar_pg)
            # loss, gradients
            value_loss = baseline_loss_scaling * 0.5 * torch.sum((v_s.view_as(values) - values)**2)
            pg_loss = compute_pg_loss(pi_logits, actions, pg_advantages)
            entropy_loss = entropy_regularizer * torch.sum(torch.softmax(pi_logits, dim=1) * torch.log_softmax(pi_logits, dim=1))
            # overall loss
            total_loss = value_loss + pg_loss + entropy_loss

            optimizer.zero_grad()

            ### learner backward time 측정 시작 ###
            start = time.perf_counter()        
            total_loss.backward()
            end = time.perf_counter()
            learner_backward_time = end - start       
            ### learner backward time 측정 종료 ###
            
            logger.info(f"value loss: {value_loss.item()}, policy gradient loss: {pg_loss.item()}, entropy: {entropy_loss.item()}, IS max: {torch.max(rhos)}, IS min: {torch.min(rhos)}, IS avg: {torch.mean(rhos)}, batching time: {learner_batching_time}, forward time: {learner_forward_time}, backward time: {learner_backward_time}, episode: {episode.value}")

            # nn.utils.clip_grad_norm_(learner_model.parameters(), clipping_grad)
            optimizer.step()

        # terminate conditon
        if episode.value >= max_episode or terminate_event.is_set():
            terminate_event.set()
            break

    logger.info(f'Learner finished, episode: {episode.value}, terminate condition : {terminate_event.is_set()}')


class IMPALA(nn.Module):

    def __init__(self, d_states, n_actions):
        super(IMPALA, self).__init__()
        
        self.d_state = d_states  # the dimension of states
        self.n_actions = n_actions  # the number of actions
        # self.lstm = use_lstm

        # fully connected
        self.fc1 = nn.Linear(self.d_state, 128) 
        self.fc2 = nn.Linear(128, 64)

        self.pi = nn.Linear(64, self.n_actions)  # policy, number of ouputs = 2 
        self.v = nn.Linear(64, 1)  #values, number of ouput = 1

        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        pi_logits = self.pi(x)
        v_theta = self.v(x)

        return pi_logits, v_theta


# hyperparameteres
n_actors = 4
max_episode = 2000
gamma = 0.99  # discount factor
baseline_loss_scaling = 0.5
entropy_regularizer = 0.008
learning_rate = 0.001

unroll_length = 30  # Actor가 수집하는 데이터 수
batch_size = 128  # Learner가 학습에 사용하는 배치 크기

clipping_rho = 2.0  # reward_clipping
clipping_c = 1.0  
clipping_rho_bar_pg = 1.0
# clipping_grad = 100.0  # grad_norm_clipping

if __name__ == "__main__":
    # main_start = time.perf_counter()
    d_states = 4  # env.observation_space.shape[0]
    n_actions = 2  # env.action_space.n

    # model
    actor_model = IMPALA(d_states, n_actions)
    
    learner_model = IMPALA(d_states, n_actions)  # learner model
    learner_model.share_memory()  # leaner_model의 parameter 공유
    optimizer = optim.RMSprop(learner_model.parameters(), lr=learning_rate, momentum=0, eps=0.01, alpha=0.99)  # epsilon, weight_decay=0.99
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    processes = []
    
    # mp.set_start_method('spawn', force=True) 
    ctx = mp.get_context("spawn")
    lock = ctx.Lock()  # Lock 생성
    queue = ctx.Queue()
    terminate_event = ctx.Event()
    
    episode = ctx.Value('i', 0)
    total_time = ctx.Value('i', 0)
    
    # actor process
    for i in range(n_actors): # 4개의 actors
        actor = ctx.Process(target=actor_process, 
                       args=(i, actor_model, learner_model, queue, unroll_length, episode, total_time, lock, terminate_event))
        actor.start()
        processes.append(actor)

    # learner process
    learner = ctx.Process(target=learner_process,
                         args=(learner_model, queue, episode, batch_size, optimizer, lock, terminate_event))
    learner.start()
    processes.append(learner)

    # terminate multiprocessing
    timeout = max_episode // 100
    for p in processes:
        if p.is_alive():
            print(f"Process {p.name} is still running")
            p.join(timeout=timeout)
            # p.join()
        else:
            print(f"Process {p.name} has exited.")
        timeout -= 4 
    for p in processes:	
        p.terminate()
    # main_end = time.perf_counter()
    # print(main_end - main_start)
    print('impala finished')