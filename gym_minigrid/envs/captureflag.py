#!/usr/bin/env python3

from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class CaptureTheFlagEnv(MiniGridEnv):
    """
    Environment for agent to capture flag and return base
    """

    def __init__(self,
        gridSize,
        lavaCount = 10,
        wallCount = 10,
        maxSteps = 100,
    ):

        self.gridSize = gridSize
        self.lavaCount = lavaCount
        self.wallCount = wallCount

        super(CaptureTheFlagEnv, self).__init__(
            gridSize=gridSize,
            maxSteps=maxSteps,
            orientation_mode=False,
            partially_observable=False,
        )

    def _genGrid(self, width, height):

        self.startPos = (
            self._randInt(0, width),
            self._randInt(0, height)
        )

        # Create the grid
        self.grid = Grid(width, height)
        wall = Wall()

        prevDoorColor = None

        # Place the flag
        while True:
            flag_pos = (
                self._randInt(0, width),
                self._randInt(0, height)
            )

            # Make sure the goal doesn't overlap with the agent
            cell = self.grid.get(*flag_pos)
            if not cell and flag_pos != self.startPos:

                self.grid.set(*flag_pos, Flag('red'))
                break

        #key_pos = self._randPos( 0, width, 0, height,)
        #self.grid.set(*key_pos, Flag('red'))

        # Place the final goal
        while True:
            self.goalPos = (
                self._randInt(0, width),
                self._randInt(0, height)
            )

            # Make sure the goal doesn't overlap with the agent
            if self.goalPos != self.startPos:
                self.grid.set(*self.goalPos, Goal())
                break

        # Placing walls
        #TODO generate great walls
        for i in range(self.wallCount):
            for i in range(self.gridSize**2): #prevent infinite loop
                #avoid placing wall on undesired places
                wall_pos = self._randPos(0, width, 0, height,)
                cell = self.grid.get(*wall_pos)
                if not cell and wall_pos != self.startPos:
                    self.grid.set(*wall_pos, Wall())
                    break

        # Placing lava
        for i in range(self.lavaCount):
            for i in range(self.gridSize**2): #prevent infinite loop
                #avoid placing lava on undesired places
                lava_pos = self._randPos(0, width, 0, height,)
                cell = self.grid.get(*lava_pos)
                if not cell and lava_pos != self.startPos:
                    self.grid.set(*lava_pos, Lava())
                    break

        self.mission = 'traverse the rooms to get to the goal'

    def tryMove(self, newPos):
        """
        Modified tryMove so that goal is only reachable when user is carrying flag
        :param newPos:
        :return:
        """
        reward = 0
        done = False

        # Boundary check
        if newPos[0] < 0 or newPos[0] >= self.gridSize \
            or newPos[1] < 0 or newPos[1] >= self.gridSize:
            return done, reward

        targetCell = self.grid.get(newPos[0], newPos[1])
        if targetCell == None:
            self.previous_cell = targetCell
            self.previous_pos = self.agentPos
            self.agentPos = newPos
        elif targetCell.type == 'goal' and isinstance(self.carrying, Flag):
            self.previous_cell = targetCell
            self.previous_pos = self.agentPos
            self.agentPos = newPos
            done = True
            reward = 1; #100 - self.stepCount
        elif targetCell.type == 'lava':
            done = True
            reward = -1000 - self.stepCount
        elif targetCell.canOverlap():
            self.previous_cell = targetCell
            self.previous_pos = self.agentPos
            self.agentPos = newPos
            if targetCell.canPickup() and self.carrying is None:
                self.carrying = targetCell
                #self.grid.set(*newPos, None)
        return done, reward

    def renderAgent(self, r):

        r.translate(
            CELL_PIXELS * (self.agentPos[0] + 0.5),
            CELL_PIXELS * (self.agentPos[1] + 0.5)
        )
        r.rotate(self.agentDir * 90)
        r.setLineColor(255, 255, 0)
        r.setColor(255, 255, 0)
        r.drawCircle(0,0,10,)
        r.setColor(0, 0, 0)
        r.drawCircle(4, -3,2,)
        r.drawCircle(-4, -3, 2, )

class CaptureTheFlagBasic(CaptureTheFlagEnv):
    def __init__(self):
        super().__init__(
            gridSize=24,
        )

class CaptureTheFlagStatic(CaptureTheFlagEnv):
    def __init__(self):
        super().__init__(
            gridSize=9,
        )

    def _genGrid(self, width, height, RANDOM_RESET):

        if RANDOM_RESET:
            self.startPos = (self._randInt(0, width/2),self._randInt(0, height));
        else:
            self.startPos = (0,height)

        # Create the grid
        self.grid = Grid(width, height)
        width -= 1
        height -= 1

        # Place the flag
        self.grid.set(*(4,0), Flag('red'))

        # Place the final goal
        self.grid.set(*(width,height), Goal())
        #self.grid.set(*(width-2, height-2), Goal())

        # Placing walls
        self.grid.set(*(4, 1), Wall())
        self.grid.set(*(4, 2), Wall())
        self.grid.set(*(4, 3), Wall())
        self.grid.set(*(4, 4), Wall())
        self.grid.set(*(4, 5), Wall())
        self.grid.set(*(4, 6), Wall())
        self.grid.set(*(4, 7), Wall())
        self.grid.set(*(4, 8), Wall())

        # Placing lava
        #self.grid.set(*(3, 3), Lava())
        #self.grid.set(*(2, 2), Lava())
        #self.grid.set(*(0, 5), Lava())
        #self.grid.set(*(3, 8), Lava())
        #self.grid.set(*(8, 3), Lava())

        self.mission = 'traverse the rooms to get to the goal'


class CaptureTheFlagTest(CaptureTheFlagEnv):
    def __init__(self):
        super().__init__(
            gridSize=9,
        )

    def _genGrid(self, width, height, RANDOM_RESET):

        if RANDOM_RESET:
            self.startPos = (self._randInt(0, width/2),self._randInt(0, height));
        else:
            self.startPos = (0,height)

        # Create the grid
        self.grid = Grid(width, height)
        width -= 1
        height -= 1

        # Place the flag
        self.grid.set(*(4,0), Flag('red'))

        # Place the final goal
        self.grid.set(*(width,height), Goal())
        #self.grid.set(*(width-2, height-2), Goal())

        # Placing walls
        self.grid.set(*(4, 1), Wall())
        self.grid.set(*(4, 2), Wall())
        self.grid.set(*(4, 3), Wall())
        self.grid.set(*(4, 4), Wall())
        self.grid.set(*(4, 5), Wall())
        self.grid.set(*(4, 6), Wall())
        self.grid.set(*(4, 7), Wall())
        self.grid.set(*(4, 8), Wall())


        # Placing lava
        self.grid.set(*(3, 3), Lava())
        #self.grid.set(*(5, 6), Lava())
        #self.grid.set(*(0, 2), Lava())
        #self.grid.set(*(0, 3), Lava())
        #self.grid.set(*(8, 3), Lava())

        self.mission = 'traverse the rooms to get to the goal'


register(
    id='MiniGrid-CaptureTheFlag-Test-v0',
    entry_point='gym_minigrid.envs:CaptureTheFlagTest',
    reward_threshold=1000.0
)


register(
    id='MiniGrid-CaptureTheFlag-Basic-v0',
    entry_point='gym_minigrid.envs:CaptureTheFlagBasic',
    reward_threshold=1000.0
)

register(
    id='MiniGrid-CaptureTheFlag-Static-v0',
    entry_point='gym_minigrid.envs:CaptureTheFlagStatic',
    reward_threshold=1000.0
)

