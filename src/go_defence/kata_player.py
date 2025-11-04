from engines.KataGo.python.katago.game.gamestate import GameState
from engines.KataGo.python.katago.train import load_model
import torch

class Kata_Player():

    colstr = 'ABCDEFGHJKLMNOPQRST'
    sgf_colstr = 'abcdefghijklmnopqrs'
    BOARD_SIZE = 19

    def __init__(self, model_name: str, model_path, extra_outputs):

        self.kata_model, kata_swa_model, other_state_dic = load_model.load_model(model_path, use_swa=True, device = torch.device("cpu"))

        self.game_state = GameState(19, GameState.RULES_TT)

        self.extra_outputs = extra_outputs

        self.white = 2
        self.black = 1

        self.colour = None
        self.opp_colour = None

        self.pass_move = b'pass'

        self.sgf = '= ;FF[4]GM[1]SZ[19]PB[]PW[]HA[0]KM[7.5]RU[TrompTaylor]RE[];'


    def play_move(self, move, colour):

        self.sgf = self.sgf + self.convert_move_sgf_format(move,colour)
        move = move.decode('utf-8')
        move = self.parse_coord(move, self.game_state.board)
        self.game_state.play(colour, move)

    def parse_coord(self, move, board):

        resign_moves = ['esign\n', b'esign\n', b'resign\n', 'resign\n', b'Resign\n', 'Resign\n']
        pass_moves = [b'pass\n', 'pass\n', b'PASS\n', 'PASS\n']

        if move in pass_moves :
            return board.PASS_LOC
        elif move in resign_moves:
            return board.PASS_LOC
        else:
            return board.loc(self.colstr.index(move[0].upper()), 19 - int(move[1:]) )

    def parse_move(self, move):
        if move == 0:
            return('pass')
        else:
            y = self.BOARD_SIZE - (int(move/(self.BOARD_SIZE + 1)) - 1)
            x = (move % (self.BOARD_SIZE + 1)) - 1
            return(self.colstr[x] + str(y))
    
    def convert_move_sgf_format(self, move, colour):

        move = move.decode('utf-8')

        colour = 'BW'[colour-1]

        resign_moves = ['esign\n', b'esign\n', b'resign\n', 'resign\n', b'Resign\n', 'Resign\n']
        pass_moves = [b'pass\n', 'pass\n', b'PASS\n', 'PASS\n', 'pass', b'pass']

        if move in pass_moves :
            sgf_move = colour +'[];'
        elif move in resign_moves:
            sgf_move = colour +'[Resign];'
        else:
            print('Move: ' + move)
            coord = (self.colstr.index(move[0].upper()), 19 - int(move[1:]) )
            sgf_move = colour+ '[' + self.sgf_colstr[coord[0]] + self.sgf_colstr[coord[1]] +'];'
        
        return(sgf_move)

    def gen_move(self, prev_move = None):

        if(prev_move != None):
            self.play_move(prev_move, self.opp_colour)

        next_move = self.game_state.get_model_outputs(self.kata_model)['genmove_result']
        self.game_state.play(self.colour, next_move)
        next_move = self.parse_move(next_move)

        self.sgf = self.sgf + self.convert_move_sgf_format(next_move.encode('utf-8'), self.colour)

        return(next_move.encode('utf-8'))
    
    def gen_final_score(self):
        score = self.game_state.get_model_outputs(self.kata_model)['scoring']

        return(score)

    def get_sgf(self):
        return(self.sgf.encode('utf-8'))
    
    def end_game(self):
        self.colour = b'w'
        self.opp_colour = b'b'
        self.sgf = '= ;FF[4]GM[1]SZ[19]PB[]PW[]HA[0]KM[7.5]RU[TrompTaylor]RE[];'
        self.game_state = GameState(19, GameState.RULES_TT)

    def get_board(self):
        board_out = self.game_state.board.to_string()
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

    def get_det_model_inputs(self):

        model_outputs = self.game_state.get_model_outputs(model=self.kata_model, extra_output_names=self.extra_outputs)
        return(model_outputs[self.extra_outputs[0]])
            


    def get_colour(self):
        return(self.colour)