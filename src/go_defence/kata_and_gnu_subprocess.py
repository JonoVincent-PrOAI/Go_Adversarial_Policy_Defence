from subprocess import Popen, PIPE, STDOUT
from go_defence.kata_gtp_subprocess import Kata_GTP_Subprocess
from go_defence.gnu_go_subprocess import GNU_Go_Subprocess
import math 

class Kata_and_GNU_Subprocess():

    def __init__(self, model_name: str, model_path, config_path, switch_point):

        self.model_name = model_name
        self.switch_point = math.ceil(switch_point / 2)
        self.num_moves = 0

        self.white = b'w'
        self.black = b'b'

        self.colour = None
        self.opp_colour = None

        self.kata_subprocess = Kata_GTP_Subprocess(model_name, model_path, config_path)
        self.GNU_subprocess = GNU_Go_Subprocess()

        self.active_GTP = self.kata_subprocess
        self.inactive_GTP = self.GNU_subprocess


    def gen_move(self, prev_move = None):

        if self.num_moves == self.switch_point:
            self.active_GTP = self.GNU_subprocess
            self.inactive_GTP = self.kata_subprocess

            print('switched to gnu')

        next_move = self.active_GTP.gen_move(prev_move)

        self.inactive_GTP.play_move(prev_move, self.opp_colour)
        self.inactive_GTP.play_move(next_move, self.colour)

        self.num_moves = self.num_moves + 1
        print(self.get_sgf())
        return(next_move)
    
    def gen_final_score(self):
        return(self.active_GTP.gen_final_score())
    
    def get_sgf(self):
        return(self.kata_subprocess.get_sgf())
    
    def get_board(self):
        return(self.active_GTP.get_board())
    
    def end_game(self):

        self.num_moves = 0
        self.active_GTP = self.kata_subprocess
        self.inactive_GTP = self.GNU_subprocess

        self.kata_subprocess.end_game()
        self.GNU_subprocess.end_game()

    def get_pass_move(self):
        return(self.active_GTP.get_pass_move())
    
    def set_colour(self, new_colour):

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
