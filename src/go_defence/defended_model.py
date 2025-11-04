import torch
from subprocess import Popen, PIPE, STDOUT
from go_defence.kata_gtp_subprocess import Kata_GTP_Subprocess
from go_defence.gnu_go_subprocess import GNU_Go_Subprocess
from go_defence.kata_player import Kata_Player
from adv_detection.adv_det_model import Model
import sys
#TODO switch switch threshold to dict

'''
---description---
Model defended using GoAD. Combines together two different subprocesses and a detection model.
---input---
model_name: str: name used to refer to the moel in print statements.
model_path: pytorch model file for the Go Model
det_model_path: pytorch model file for detection model
switch threshold: 2d list: 
    first element is number of adv classifications needed to trigger detection
    second element is the number of past classifications which is stored
accurate_window: cut off point where deteciton is now longer allowed 
'''
class Defended_Model():

    def __init__(self, model_name: str, model_path, model_chkpt_path, det_model_path, switch_threshold, accurate_window, config):
        self.model_name = model_name
        self.switch_threshold = switch_threshold
        self.num_moves = 0

        self.white = b'w'
        self.black = b'b'

        self.colour = None
        self.opp_colour = None

        self.GNU_subprocess = GNU_Go_Subprocess()
        self.adv_det_model = torch.load(det_model_path, weights_only=False)
        self.adv_det_model.eval()
        self.kata_model = Kata_Player('may_24', model_chkpt_path, ['rconv18.out'])
        self.kata_subprocess = Kata_GTP_Subprocess('GoAD-kata', model_path, config)

        self.det_model_pred = []
        self.num_stored = switch_threshold[1]
        self.threshold = switch_threshold[0]
        self.delay = accurate_window[0]
        self.end_point = accurate_window[1]

        self.switch_point = None

        self.active_GTP = self.kata_subprocess
        self.inactive_GTP = self.GNU_subprocess

    '''
    ---description---
    Generates a GTP move in the current board state. If called when it is not the models turn it will generate an 
    illegal move.
    ---input---
    prev_move = None: the opponents last move. If left as none presumes first player and sets colour as black.
    ---output---
    next_move: GTP move. 
    #TODO reduce the scope of this mehtod if possible.
    ===Warning====
    This method has lots of side effects. Prev_move and next_move are both used to update the board state for 
    models and prev_move is used to determine the players colour.
    '''
    def gen_move(self, prev_move = None):

        with torch.no_grad():

            input = self.kata_model.get_det_model_inputs()
            input = torch.tensor(input)
            output = self.adv_det_model(input)[0]

            if output[0] > output[1]:
                pred = 0
            else:
                pred = 1
            
            self.det_model_pred.append(pred)

        if len(self.det_model_pred) > self.num_stored:
            self.det_model_pred.pop(0)

        if self.num_moves > self.delay:
            if self.num_moves < self.end_point:
                if sum(self.det_model_pred) >= self.threshold:
                    if self.switch_point == None:

                        self.active_GTP = self.GNU_subprocess
                        self.inactive_GTP = self.kata_subprocess
                    
                        self.switch_point = self.num_moves
                        print('switched to gnu')

        next_move = self.active_GTP.gen_move(prev_move)

        
        self.inactive_GTP.play_move(prev_move, self.inactive_GTP.opp_colour)
        self.inactive_GTP.play_move(next_move, self.inactive_GTP.colour)

        self.kata_model.play_move(prev_move, self.kata_model.opp_colour)
        self.kata_model.play_move(next_move, self.kata_model.colour)

        self.num_moves = self.num_moves + 2
        return(next_move)
    
    '''
    ---description---
    Returns GNU models calualtion of the final score
    '''
    def gen_final_score(self):
        return(self.GNU_subprocess.gen_final_score())
    
    def get_sgf(self):
        return(self.kata_subprocess.get_sgf())
    
    def get_board(self):
        return(self.active_GTP.get_board())
    
    '''
    ---description---
    Resets important variables and should be called whenever a game ends.
    '''
    def end_game(self):
        with open('src/go_attack/Experiment_Results/defended-switch-points.txt','a' ) as f:
            f.write(str(self.switch_point) + ",")

        self.switch_point = None

        self.num_moves = 0
        self.active_GTP = self.kata_subprocess
        self.inactive_GTP = self.GNU_subprocess

        self.kata_model.end_game()
        self.kata_subprocess.end_game()
        self.GNU_subprocess.end_game()

    def get_pass_move(self):
        return(self.active_GTP.get_pass_move())
    
    def set_colour(self, new_colour):

        self.kata_model.set_colour(new_colour)
        self.kata_subprocess.set_colour(new_colour)
        self.GNU_subprocess.set_colour(new_colour)

        if new_colour == 'b':
            self.colour = self.black
            self.opp_colour = self.white
        elif new_colour == 'w':
            self.colour = self.white
            self.opp_colour = self.black

    def get_colour(self):
        return(self.active_GTP.get_colour())
