from subprocess import Popen, PIPE, STDOUT
from go_defence.kata_gtp_subprocess import Kata_GTP_Subprocess

class Subprocess_Game():

    def __init__(self, model1 : Kata_GTP_Subprocess, model2 : Kata_GTP_Subprocess):

        self.pla = model1
        self.opp = model2

        self.pla_move = None
        self.opp_move = None

    
    def game(self, print_mode : bool):

        self.pla.set_colour('b')
        self.opp.set_colour('w')

        self.pla_move = self.pla.gen_move()
        self.opp_move = self.opp.gen_move(self.pla_move)

        if print_mode:
            print('pla board:')
            print(self.pla.get_board())
            print()
            print('opp board: ')
            print(self.opp.get_board())

            print()

        while not(self.is_game_over()):

            self.pla_move = self.pla.gen_move(self.opp_move)
            if self.pla_move != b'resign\n':
                self.opp_move = self.opp.gen_move(self.pla_move)

            print(str(self.pla.get_colour()) + ': ' + str(self.pla_move))
            print(str(self.opp.get_colour()) + ': ' + str(self.opp_move))

            if print_mode:
                print('pla board:')
                print(self.pla.get_board())
                print()
                print('opp board: ')
                print(self.opp.get_board())
                print()
        result = self.pla.gen_final_score().decode('utf-8')
        sgf = self.pla.get_sgf().decode('utf-8')
        sgf = sgf.replace('PB[]', 'PB[' + self.pla.model_name + ']')
        sgf = sgf.replace('PW[]', "PW[" + self.opp.model_name + ']')
        self.end_game()
        return(result, sgf)

    def is_game_over(self):

        resign_moves = ['esign\n', b'esign\n', b'resign\n', 'resign\n']

        if (self.pla_move == self.pla.get_pass_move() and self.opp_move == self.opp.get_pass_move()):
            return(True)
        elif self.pla_move in resign_moves:
            return(True)
        elif self.opp_move in resign_moves:
            return(True)
        else:
            return(False)
        
    def end_game(self):
        self.pla.end_game()
        self.opp.end_game()
        
        temp = self.pla
        self.pla = self.opp
        self.opp = temp
        temp = None

        self.pla_move = None
        self.opp_move = None