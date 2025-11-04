from subprocess import Popen, PIPE, STDOUT


'''
---description---
A subprocess which runs GnuGO using it's GTP interface.
'''
class GNU_Go_Subprocess():

    def __init__(self):

        self.model_name = 'GnuGO'

        command = 'gnugo'
        subcommand = '--mode' 
        sub_arg ='gtp'

        #these enforce the tromp-taylor rules
        score = '--score'
        score_arg = 'aftermath'
        arg2 = '--capture-all-dead'
        arg3 = '--chinese-rules'
        suicide_arg = '--allow-suicide'


        self.full_command = [command, subcommand, sub_arg, score, score_arg, arg2, arg3, suicide_arg]

        self.white = b'white'
        self.black = b'black'

        self.colour = None
        self.opp_colour =None

        self.GNU_Go_GTP = None

        self.showboard_size = 24
        self.pass_move = b'PASS\n'

    '''
    ---description---
    Generates the next the GnuGO player will take.
    ---input---
    prev_move = None: The previous move of the Gnu Players' opponent. If left as Nones asssumes it is the first player in
                     a new game
    ---output---
    next_move: the next ove of the GNU player

    ===WARNING===
    this method has lots of side effects. Both next and prev moves will effect the internal boardstate of the
    GNU player.
    '''
    def gen_move(self, prev_move = None):

        if self.GNU_Go_GTP == None:
            self.GNU_Go_GTP = Popen(self.full_command, stdout=PIPE, stdin=PIPE, stderr=PIPE)

        if(prev_move != None):
            self.play_move(prev_move, self.opp_colour)

        self.GNU_Go_GTP.stdin.write(b'genmove ' + self.colour + b'\n')
        self.GNU_Go_GTP.stdin.flush()
        next_move = self.GNU_Go_GTP.stdout.readline()

        while next_move == b'= \n' or next_move == b'\n':
            next_move = self.GNU_Go_GTP.stdout.readline()

        next_move = next_move.decode('utf-8').replace('= ', '')

        return(next_move.encode('utf-8'))
    
    '''
    ---description---
    Plays a move so that GNUs' internal boardstate is updated and aligns with the true board state
    ---input---
    move: the position of the mvoe
    colour: the colour of the player playing the move
    '''
    def play_move(self, move, colour):

        if self.GNU_Go_GTP == None:
            self.GNU_Go_GTP = Popen(self.full_command, stdout=PIPE, stdin=PIPE, stderr=PIPE)

        if move != None:
            self.GNU_Go_GTP.stdin.write(b'play ' + colour + b' ' + move +b'\n')
    
    def get_board(self):
        self.GNU_Go_GTP.stdin.write(b'showboard\n')
        self.GNU_Go_GTP.stdin.flush()
        board_out = ''
        for _ in range(self.showboard_size):
            board_out = board_out + (self.GNU_Go_GTP.stdout.readline().decode('utf-8'))

        return(board_out)
    
    def clear_board(self):
        self.GNU_Go_GTP.stdin.write(b'clear_board')
        self.GNU_Go_GTP.stdin.flush()

    def clear_cache(self):
        self.GNU_Go_GTP.stdin.write(b'clear_cache')
        self.GNU_Go_GTP.stdin.flush()

    '''
    ---description---
    Resets important variables and should be called at the end of every game.
    '''
    def end_game(self):
        self.clear_cache()
        self.clear_board()
        self.colour = b'white'
        self.opp_colour = b'black'
        self.GNU_Go_GTP.terminate()
        self.GNU_Go_GTP = None

    def gen_final_score(self):
        self.GNU_Go_GTP.stdin.write(b'final_score\n')
        self.GNU_Go_GTP.stdin.flush()
        score = self.GNU_Go_GTP.stdout.readline()
        while score =='\n':
            score = self.GNU_Go_GTP.stdout.readline()
        return(score)
    
    def get_sgf(self):
        
        self.GNU_Go_GTP.stdin.write(b'printsgf\n')
        self.GNU_Go_GTP.stdin.flush()
        self.GNU_Go_GTP.stdout.readline()
        sgf = self.GNU_Go_GTP.stdout.readline()
        self.GNU_Go_GTP.stdout.readline()

        return(sgf + b'GNUs SGF Does not work :(')
    
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