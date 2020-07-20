import gym
from rlkit.envs.sawyer_hammer import SawyerHammerEnv
import imageio
import os

variant = {
                    'obs_type': 'rgbd',
                    'assets_path': '/h/lichothu/hammers/assets/',
                    'task_id': 4,
                    'num_kp': 8,
                    'random_hammer_zrot': True,
                    'rotMode': 'rotz'
}

frames = []
env = SawyerHammerEnv(**variant)
# os.mkdir('/checkpoint/lichothu/$SLURM_JOB_ID')
video_writer = imageio.get_writer(('/checkpoint/lichothu/test.mp4'), fps=20)
for i_episode in range(1):
    observation = env.reset()
    for t in range(1000):
        img = env.sim.render(height=512, width=512, camera_name='camera_grasp')[::-1]
        frames.append(img)
        video_writer.append_data(img)
        # env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

video_writer.close()
env.close()