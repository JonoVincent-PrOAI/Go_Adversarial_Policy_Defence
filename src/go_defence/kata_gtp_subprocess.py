from subprocess import Popen, PIPE, STDOUT
import sys


class Kata_GTP_Subprocess():

    def __init__(self, model_name: str, model_path, config_path):
        if '../../' not in sys.path:
            sys.path.append('../../')
        
        self.model_name = model_name
        command = 'katago'
        subcommand = 'gtp'
        model_arg = '-model ' + model_path
        config_arg = '-config ' + config_path

        self.full_command = [command, subcommand, model_arg, config_arg]

        self.white = b'w'
        self.black = b'b'

        self.colour = None
        self.opp_colour = None

        self.kata_GTP = None

        self.showboard_size = 26
        self.pass_move = b'pass\n'

    def gen_move(self, prev_move = None):

        if self.kata_GTP == None:
            self.kata_GTP = Popen(self.full_command, stdout=PIPE, stdin=PIPE, stderr=PIPE)

        
        if(prev_move != None):
            self.play_move(prev_move, self.opp_colour)

        self.kata_GTP.stdin.write(b'genmove ' + self.colour + b'\n')
        self.kata_GTP.stdin.flush()
        next_move = self.kata_GTP.stdout.readline()

        while next_move == b'= \n' or next_move == b'\n':
            next_move = self.kata_GTP.stdout.readline()

        next_move = next_move.decode('utf-8').replace('= ', '')

        return(next_move.encode('utf-8'))
    

    def play_move(self, move, colour):

        if self.kata_GTP == None:
            self.kata_GTP = Popen( self.full_command, stdout=PIPE, stdin=PIPE, stderr=PIPE)

        if(move != None):
            self.kata_GTP.stdin.write(b'play ' + colour + b' ' + move + b'\n')
            self.kata_GTP.stdin.flush()
            self.kata_GTP.stdout.readline()

    
    def gen_final_score(self):
        self.kata_GTP.stdin.write(b'final_score\n')
        self.kata_GTP.stdin.flush()
        self.kata_GTP.stdout.readline()
        score = self.kata_GTP.stdout.readline()
        while score == b'\n' or score == b'= \n':
            score = self.kata_GTP.stdout.readline()

        return(score)

    def get_sgf(self):
        self.kata_GTP.stdin.write(b'printsgf\n')
        self.kata_GTP.stdin.flush()
        sgf = self.kata_GTP.stdout.readline()
        while sgf == b'= \n' or sgf == b'\n':
            sgf = self.kata_GTP.stdout.readline()

        return(sgf)
    
    def clear_board(self):
        self.kata_GTP.stdin.write(b'clear_board')
        self.kata_GTP.stdin.flush()

    def clear_cache(self):
        self.kata_GTP.stdin.write(b'clear_cache')
        self.kata_GTP.stdin.flush()

    def end_game(self):
        self.clear_cache()
        self.clear_board()
        self.colour = b'w'
        self.opp_colour = b'b'
        self.kata_GTP.terminate()
        self.kata_GTP = None

    def get_board(self):
        self.kata_GTP.stdin.write(b'showboard\n')
        self.kata_GTP.stdin.flush()
        board_out = ''
        for _ in range(self.showboard_size):
            board_out = board_out + (self.kata_GTP.stdout.readline().decode('utf-8'))

        return(board_out)

    
    def get_pass_move(self):
        return(self.pass_move)
    
    def set_colour(self, colour):
        if colour == 'b':
            self.colour = self.black
            self.opp_colour = self.white
        elif colour == 'w':
            self.colour = self.white
            self.opp_colour = self.black

    def get_colour(self):
        return(self.colour)