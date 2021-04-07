from typing import Tuple

import numpy as np
import math
from ..miniworld import MiniWorldEnv
import matplotlib.pyplot as plt
from ..entity import Box
from ..params import DEFAULT_PARAMS
from gym import spaces

class OneRoom(MiniWorldEnv):
    """
    Environment in which the goal is to go to a red box
    placed randomly in one big room.
    """

    def __init__(self, size=10, max_episode_steps=180, num_agents=0, pref_speed=1.3, max_speed=1.5, delta_time=0.10, **kwargs):
        assert size >= 2
        self.size = size
        self.pref_speed = pref_speed
        self.max_speed = max_speed
        self.delta_time = delta_time

        self.num_agents = num_agents
        self.agents = []

        super().__init__(
            max_episode_steps=max_episode_steps,
            **kwargs
        )

        # Drive with a linear and turning force. This is pretty much polar velocity.
        self.action_space = spaces.Box(low=np.array([0, -np.pi / 2]),
                                       high=np.array([max_speed, np.pi / 2]), dtype=np.float32)

        # Observe with a depth image 60x80 and the relative position of the goal
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=(60, 80, 1)),
            spaces.Box(low=0, high=size, shape=(2,))
        ))

    def _gen_world(self):
        room = self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size
        )

        self.goal = self.place_entity(Box(color='red', size=0.5, render=True))

        # Add all agents to the world
        for _ in range(self.num_agents):
            self.agents.append(ORCAAgent(self.size, self.size))
            # self.agents.append(ORCAAgent((0, 0)))
            self.place_entity(self.agents[-1])
            self.agents[-1].added_to_env()

        # Place the agent down
        self.agent = self.place_agent()
        self.orca_num = ORCAAgent.add_agent_to_orca(self.agent)

    def step(self, action: Tuple[float, float]):
        # Get the linear and rotational aspects of the action
        linear = action[0]
        rot = action[1]

        # Turn this into an X and Y velocity
        self.agent.dir += rot
        self.agent.dir %= 2 * np.pi
        vel = np.array([np.sin(self.agent.dir + np.pi / 2), np.cos(self.agent.dir + np.pi / 2)]) * self.max_speed * linear
        vel = np.array([vel[0], 0, vel[1]])


        # Keep old position for reward
        old_pos = self.agent.pos

        # Check that the new position does not intersect with anything
        prop_position = self.agent.pos + vel * self.delta_time
        if not self.intersect(self.agent, prop_position, self.agent.radius):
            self.agent.pos = prop_position

        # Step the agent's simulation
        ORCAAgent.ORCA.doStep()

        # Step all of the other agents
        for agent in self.agents:
            agent.step(self.delta_time)


        # Find the relative position of the goal and return that on top of the depth image
        goal_vec = np.array(self.goal.pos) - np.array(self.agent.pos)
        goal_vec = np.array(goal_vec[0], goal_vec[2])

        # Update the ego agent's position in the ORCA simulation
        ORCAAgent.ORCA.setAgentPosition(self.orca_num, (self.agent.pos[0], self.agent.pos[2]))
        ORCAAgent.ORCA.setAgentVelocity(self.orca_num, (vel[0], vel[2]))

        obs = np.clip(self.render_depth(), 0, 4.5)
        done = False
        reward = 0

        self.step_count += 1
        if self.step_count >= self.max_episode_steps:
            done = True
            reward = 0
            return obs, reward, done, {"term_reason": "max_steps"}

        if any([self.near(agent, eps=0.05) for agent in self.agents]):
            reward = -10
            done = True
            return obs, reward, done, {"term_reason": "agent_collision"}

        if self.near(self.goal):
            reward += 10
            return obs, reward, True, {"term_reason": "goal"}

        old_goal_dist = np.linalg.norm(self.goal.pos - old_pos)
        new_goal_dist = np.linalg.norm(self.goal.pos - self.agent.pos)
        reward += (old_goal_dist - new_goal_dist) / np.linalg.norm(vel)

        return (obs, goal_vec), reward, done, {}

class OneRoomS6(OneRoom):
    def __init__(self, max_episode_steps=100, **kwargs):
        super().__init__(size=6, max_episode_steps=max_episode_steps, **kwargs)

class OneRoomS6Fast(OneRoomS6):
    def __init__(self, forward_step=0.7, turn_step=45):
        # Parameters for larger movement steps, fast stepping
        params = DEFAULT_PARAMS.no_random()
        params.set('forward_step', forward_step)
        params.set('turn_step', turn_step)

        super().__init__(
            max_episode_steps=50,
            params=params,
            domain_rand=False
        )
