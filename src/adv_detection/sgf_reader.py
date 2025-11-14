import sys
sys.path.append('../../')



import engines.KataGo.python.katago.train.load_model as load_model
import torch
from engines.KataGo.python.katago.game.board import Board
from engines.KataGo.python.katago.game.gamestate import GameState
from engines.KataGo.python.katago.game.features import Features

class SGFReader: 

    #Used to convert SGF coordinates into Board coordinates
    colstr = 'ABCDEFGHIJKLMNOPQRST'

    """
    ---description---
    function(list[str]) -> list[list[str]]
    Converts the list of moves extracted from SGF into an array of players and move co-ords
    """
    def parse_moves(moves):
        parsed_move = []
        for move in moves:
            if move != '':
                split_move = move.split('[')
                player = split_move[0]
                pos = split_move[1].replace(']', '')
                parsed_move.append([player, pos])
        
        return(parsed_move)
    
    """
    ---description---
    function(str) -> str,str,str,str
    Takes in the game metadtat from SGFS and extracts the relevant data:

    ---outputs---
    board_size: the size of board the game was played on.
    black_player: the given name of the model controlling the black player
    white_player: the given name of the model controlling the white player
    rules: the variant of the rules being used during the game 
    """
    def parse_metadata_sgfs(metadata : str):

        split_data = metadata.split(']')

        board_size = split_data[2].split('[')[1]
        black_player = split_data[3].split('[')[1]
        white_player = split_data[4].split('[')[1]
        rules = split_data[10].split('[')[1]

        return(board_size, black_player, white_player, rules)
    

    def parse_metadata_sgf(metadata : str):

        split_data = metadata.split(']')

        board_size = split_data[2].split('[')[1]
        black_player = split_data[3].split('[')[1]
        white_player = split_data[4].split('[')[1]
        rules = split_data[7].split('[')[1]

        return(board_size, black_player, white_player, rules)
    """
    ---description---
    function(str) -> list[dict]
    reads a SGF file and separates the file into separate games. Each games is then stored in a dict

    ---input---
    file_path: str: the file path to the SGF file being read

    ---output---
    sgf_game_list: list of dict, each dict contains the data on a game
        ['sz']: the board size
        ['pla']: the black player model name
        ['opp']: the the white player model name
        ['rules']: the rules used during the game
        ['moves']: list of moves made by each player. Each move is the player color and co-ord
        ['result]; the game result, wining player and score diff

    """
    def read_file(self, file_path : str):

        with open(file_path) as sgf_file:
            sgf_games = sgf_file.read().split('(')
        
        sgf_games.pop(0)
        sgf_game_list = []

        for game in sgf_games:

            game_lines = game.split(';')
            game_metadata = game_lines[1]
            game_moves_confidence = game_lines[2:]
            result = game_lines[-1].split('C')[1]
            game_moves = []

            for move_confidence in game_moves_confidence:
                game_moves.append(move_confidence.split('C')[0])


            if file_path.split(".")[-1] == 'sgfs':
                board_size, black_player, white_player, rules = self.parse_metadata_sgfs(game_metadata)
            elif file_path.split(".")[-1] == 'sgf':
                board_size, black_player, white_player, rules = self.parse_metadata_sgf(game_metadata)
            else:
                print('error: unknonw file extension, file must be .sgf or .sgfs')
            moves = self.parse_moves(game_moves)

            game_dict = {}
            game_dict['sz'] = board_size
            game_dict['pla'] = black_player
            game_dict['opp'] = white_player
            game_dict['rules'] = rules
            game_dict['moves'] = moves
            game_dict['result'] = result

            sgf_game_list.append(game_dict)
        
        return(sgf_game_list)
    
    """
    ---description---
    procedure
    replays the games in an SGF file, by passing them through a NN model

    ---input---
    file_path: str: the file path to the SGF file being replayed
    nn_chkpt: str: the file path to the chkpt file of the nn model being used pass the moves through
    """
    def replay_file(self, file_path : str, nn_chkpt : str ):

        sgf_game_list = self.read_file(SGFReader, file_path)

        kata_model, kata_swa_model, other_state_dict = load_model.load_model(nn_chkpt, use_swa=True, device = torch.device("cpu"))

        for game_dict in sgf_game_list:

            board_size = int(game_dict['sz'])

            gs = GameState(board_size, GameState.RULES_TT)

            kata_model.eval()

            for move in game_dict['moves']:
                
                command = [['play'], move[0], move[1]]
                pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
                loc = self.parse_coord(SGFReader, command[2], gs.board)
                gs.play(pla,loc)

                gs.get_model_outputs(kata_model)
            

    """
    ---description---
    procedure
    replays a game from a list of the moves, and the board size, by passing them through a NN model

    ---input---
    moves: list: list containing the moves perforemd during the game, each move is in the sgfs format
    board_size: int: the size of board the game was played on
    nn_chkpt: str: the file path to the chkpt file of the nn model being used pass the moves through
    """
    def replay_game(self, moves : list, board_size : int, nn_chkpt : str):

        kata_model, kata_swa_model, other_state_dict = load_model.load_model(nn_chkpt, use_swa=True, device = torch.device("cpu"))

        gs = GameState(board_size, GameState.RULES_TT)

        kata_model.eval()

        for move in moves:
            
            command = [['play'], move[0], move[1]]

            print(command)
            print()

            pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
            loc = self.parse_coord(SGFReader, command[2], gs.board)
            gs.play(pla,loc)

            print(gs.board.to_string())

    
    def play_move(self, gs : GameState, move : str,):

        command = [['play'], move[0], move[1]]

        pla = (Board.BLACK if command[1] == "B" or command[1] == "b" else Board.WHITE)
        loc = self.parse_coord(SGFReader, command[2], gs.board)
        gs.play(pla,loc)

    def parse_coord(self, s, board):
        if s == '':
            return board.PASS_LOC
        return board.loc(self.colstr.index(s[0].upper()), 18 - self.colstr.index(s[1].upper()))
