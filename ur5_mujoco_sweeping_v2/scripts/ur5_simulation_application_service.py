import numpy as np
from ur5_simulation_environment import UR5SimulationEnvironment
from config import Config


class UR5SimulationApplicationService:
    def run(self):
        config = Config()
        env = UR5SimulationEnvironment(config)
        step = 300

        qpos = np.zeros(env.model.nq)
        qvel = np.zeros(env.model.nv)

        # env.reset_env(qpos_init=None, qvel_init=[0] * qvel, target_position_init=[0] * qpos)
        env.reset_env(qpos_init=np.array([-0.165, 0.314, 0, 1.08, 0, -0.0314, 0.033, 0.033, -0.1, 0.8, 0.03, 1, 0, 0, 0]), qvel_init=[0] * qvel, target_position_init=[0] * qpos)

        # env.set_difference_control_input(np.array([0.3] * 6))
        for index_step in range(step):
            env.render()
            env.step()


if __name__ == '__main__':
    sim = UR5SimulationApplicationService()
    sim.run()
