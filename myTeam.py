# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
import math
import time
from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########
class mikiAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.comida_en_barriga = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        
        # You can profile your evaluation time by uncommenting these lines
        #start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        #print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if self.get_capsules(game_state):
            if self.get_maze_distance(game_state.get_agent_position(self.index), min(self.get_capsules(game_state))) < 3:
                best_dist = 9999
                best_action = None
                for action in actions:
                    successor = self.get_successor(game_state, action)
                    pos2 = successor.get_agent_position(self.index)
                    dist = self.get_maze_distance(min(self.get_capsules(game_state)), pos2)
                    if dist < best_dist:
                        best_action = action
                        best_dist = dist
                return best_action
        
        if not game_state.get_agent_state(self.index).is_pacman:
            self.comida_en_barriga = 0

        if self.get_previous_observation():
            if len(self.get_food(self.get_current_observation()).as_list()) < len(self.get_food(self.get_previous_observation()).as_list()):
                self.comida_en_barriga += 1
       
        if self.comida_en_barriga > 5:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights


    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        indices = self.get_opponents(game_state)

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(mikiAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        capsules_list = self.get_capsules(successor)
          # self.getScore(successor)
        
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        attackers = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None and not game_state.get_agent_state(self.index).is_pacman and self.index]
        
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # Compute distance to the nearest food
        features['successor_score'] = -len(food_list)
        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        """
        if len(capsules_list) > 0:
            if self.get_maze_distance(my_pos, capsules_list[0]) < 3:
                features['successor_score'] = 0
                features['distance_to_food'] = 0
                features['distance_to_capsules'] = self.get_maze_distance(my_pos, capsules_list[0])
        """
        if action == Directions.STOP: features['stop'] = 1
        #rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        #if action == rev: features['reverse'] = 1

        if self.comida_en_barriga > 2:
            features['back_home'] = self.get_maze_distance(self.start, my_pos) * self.comida_en_barriga

        features['successor_score'] = -len(food_list)
        if len(attackers) > 0:
            for a in attackers:
                if self.get_maze_distance(my_pos, a.get_position()) < 5 and a.scared_timer < 1:
                    if self.get_maze_distance(my_pos, a.get_position()) == 0:
                        features['successor_score'] = 0
                        features['distance_to_food'] = 0
                        features['distance_to_capsules'] = 0
                        features['corre'] = 100
                    if self.get_maze_distance(my_pos, a.get_position()) == 1:
                        features['distance_to_food'] = 0
                        features['distance_to_capsules'] = 0
                        features['corre'] = 1
                    if self.get_maze_distance(my_pos, a.get_position()) == 2:
                        features['attacker_distance'] = -100
                    if self.get_maze_distance(my_pos, a.get_position()) == 3:
                        features['attacker_distance'] = -10
                    if self.get_maze_distance(my_pos, a.get_position()) == 4:
                        features['attacker_distance'] = -1

        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1000,'corre': -10000000,'distance_to_food': -1,'distance_to_capsules': -20 , 'back_home': -2,'invader_distance': -20, 'attacker_distance': 10, 'stop': -100}


class DefensiveReflexAgent(mikiAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        x = game_state.data.layout.width
        y = game_state.data.layout.height   

        if game_state.is_on_red_team(self.index):
            features['go_midle'] = self.get_maze_distance(my_pos, (math.floor((x-1)/2), math.ceil((y-1)/2)))
        if not game_state.is_on_red_team(self.index):
            features['go_midle'] = self.get_maze_distance(my_pos, (math.ceil((x-1)/2), math.ceil((y-1)/2)))
       
        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None and self.index]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if my_state.scared_timer:
            if len(invaders) > 0:
                for a in invaders:
                    if self.get_maze_distance(my_pos, a.get_position()) <= 1:
                        features['num_invaders'] = 0
                        features['scared_dist'] = 1

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'scared_dist': -1000000, 'invader_distance': -10, 'stop': -100, 'reverse': -20, 'go_midle': -1}
