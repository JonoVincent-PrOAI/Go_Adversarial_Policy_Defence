
import os
from adv_detection.sgf_reader import SGFReader
import pandas as pd 
import engines.KataGo.python.katago.train.load_model as load_model
import torch
from engines.KataGo.python.katago.game.gamestate import GameState
from engines.KataGo.python.katago.game.board import Board
import json
import numpy as np
from pathlib import Path
import random


'''
---description---
Creates the dataset used to train detection model from SGF files of games. Replays games and stores the outputs 
of selected conv blocks. Also stores meta data on the games.
'''    
class Probe():

    game_header = ['black name','white name','black model', 'white model', 'adversarial', 'attack type', 'moves', 'result', 
        'file path']
    
    probe_header = ['probe model','game id','move number','black name','white name','black model', 'white model', 'adversarial', 'attack type', 'result', 
        'file path', 'layer outputs']

    non_adv_matchups = [['base-adv', 'may23-vic'],['base-adv', 'dec23'],['base-adv', 'may24'],['base-adv','v9'],
                        ['base-adv', 'ViT-Vic'],['cont','may24']]

    #name of hidden files in the data whicih need to be ignored when iterating over the directory
    cursed_files = ['.DS_Store', 'game.dat']
    #Used to map the different version of models to a single name for simplicity
    name_mapping = {'a9' : ['r9-v600'], 
                    'atari' : ['attack-h9-s564026112-v600-AMCTS','attack-h9-s564026112-v128-AMCTS',
                               'attack-h9-s564026112-v32-AMCTS', 'attack-h9-s564026112-v48-AMCTS',
                               'attack-h9-s564026112-v24-AMCTS', 'attack-h9-s564026112-v1-AMCTS', 
                               'attack-h9-s564026112-v16-AMCTS', 'attack-h9-s564026112-v256-AMCTS',
                               'attack-h9-s564026112-v64-AMCTS', 'attack-h9-s564026112-v4-AMCTS',
                               'attack-h9-s564026112-v32-AMCTS', 'attack-h9-s564026112-v48-AMCTS',
                               'attack-h9-s564026112-v24-AMCTS', 'attack-h9-s564026112-v1024-AMCTS', 
                               'attack-h9-s564026112-v2048-AMCTS', 'attack-h9-s564026112-v2-AMCTS',
                               'attack-h9-s564026112-v4096-AMCTS', 'attack-h9-s564026112-v8-AMCTS'],
                    'base-adv' : ['cyclic-v600','adv-s545065216-v600','r0-v600','adv-s545065216-v600-AMCTS-S++'],
                    'cont' : ['cont-s630m'],
                    'gift' : ['attack-b18-s651m-v256','attack-b18-s651m','attack-b18-s651m-v16',
                              'attack-b18-s651m-v64', 'attack-b18-s651m-v1', 'attack-b18-s651m-v4',
                              'attack-b18-s651m-v2','attack-b18-s651m-v128','attack-b18-s651m-v8',
                              'attack-b18-s651m-v2048', 'attack-b18-s651m-v32', 
                              'attack-b18-s651m-v4096','attack-b18-s651m-v1024'],
                    'large' : ['large-s216m'],
                    'stall' : ['attack-ft-h9-s97114624-v16-AMCTS','attack-ft-h9-s97114624-v32-AMCTS',
                               'attack-ft-h9-s97114624-v8-AMCTS', 'attack-ft-h9-s97114624-v2048-AMCTS',
                                'attack-ft-h9-s97114624-v1-AMCTS','attack-ft-h9-s97114624-v4096-AMCTS',
                                'attack-ft-h9-s97114624-v128-AMCTS', 'attack-ft-h9-s97114624-v2-AMCTS',
                                'attack-ft-h9-s97114624-v1024-AMCTS','attack-ft-h9-s97114624-v256-AMCTS',
                                'attack-ft-h9-s97114624-v4-AMCTS', 'attack-ft-h9-s97114624-v64-AMCTS',
                                'attack-ft-h9-s97114624-v600-AMCTS'],
                    'ViT-adv' : ['attack-vit-s326m'],
                    'v9' : ['h9-v512','h9-v1024','h9-v64','h9-v8192','h9-v2048','h9-v4096','h9-v2','h9-v4',
                            'h9-v256','h9-v1','h9-v8','h9-v128','h9-v16','h9-v32','h9-v16384','h9-v32768'],
                    'dec23' : ['b18-s8527m-v512','b18-s8527m-v2048','b18-s8527m-v8','b18-s8527m-v1024',
                               'b18-s8527m-v4096','b18-s8527m-v8192','b18-s8527m-v2','b18-s8527m-v32',
                               'b18-s8527m-v128','b18-s8527m-v64','b18-s8527m-v1','b18-s8527m-v2',
                               'b18-s8527m-v32','b18-s8527m-v128','b18-s8527m-v64', 'b18-s8527m-v1',
                               'b18-s8527m-v4','b18-s8527m-v16','b18-s8527m-v256'],
                    'may23-vic' : ['b60-s7702m-v4096'],
                    'may24' : ['b18-s9997m-v256','b18-s9997m-v4096','b18-s9997m-v8192','b18-s9997m-v16384',
                               'b18-s9997m-v2048', 'b18-s9997m-v8', 'b18-s9997m-v32', 'b18-s9997m-v2',
                               'b18-s9997m-v128','b18-s9997m-v4','b18-s9997m-v1024','b18-s9997m-v64', 
                               'b18-s9997m-v16','b18-s9997m-v512', 'b18-s9997m-v1','b18-s9997m-v32768',
                               'b18-s9997m-v65536'],
                    'base-vic' : ['cp505-v1000000','cp505h-v2','cp505h-v16','cp505h-v128','cp505h-v4096',
                                  'cp505h-v8192', 'cp505h-v1', 'cp505h-v1024', 'cp505h-v4', 'cp505h-v2048',
                                  'cp505h-v32','cp505h-v32768','cp505h-v256','cp505h-v512','cp505h-v16384', 
                                  'cp505h-v64', 'cp505h-v8'],
                    'ViT-Vic' : ['vit-b16-s650m-v1024','vit-v1024','vit-b16-s650m-v128','vit-b16-s650m-v4',
                                 'vit-b16-s650m-v64', 'vit-b16-s650m-v16', 'vit-b16-s650m-v256','vit-b16-s650m-v8',
                                 'vit-b16-s650m-v2','vit-b16-s650m-v32','vit-b16-s650m-v1', 'vit-b16-s650m-v512',
                                 'vit-v512','vit-v64','vit-v8','vit-v2048','vit-v8192','vit-v65536','vit-v128',
                                 'vit-v4096', 'vit-v256', 'vit-v32', 'vit-v1', 'vit-v16','vit-v4','vit-v2']
                    }

    def __init__(self, adv_dir_path, non_adv_dir_path, seed):
            
            random.seed(seed)
            
            self.non_adv_dir_path = non_adv_dir_path

            self.adv_dir_path = adv_dir_path


    '''
    ---description---
    function() -> pandas.dataframe()
    Main function of the class, calls other functions to extract the data from files, then compiles data into a
    Pandas dataframe.

    ---output---
    dataframe: pandas Dataframe containing information extracted from sgfs files
    '''
    def load_game_data(self):

        non_adv_data = self.load_non_adv_model_data(self)

        adv_data = self.load_adv_model_data(self)

        data = non_adv_data + adv_data

        dataframe = pd.DataFrame(data, columns = self.game_header)

        return(dataframe)

    '''
    ---desription---
    function() -> list(str)
    extracts the information about games in the non-adversarial files. Several fields are the same for all
    all non-adv games, so it's quickest to gve them their own method.

    ---output---
    non_adv_data: the info of each game contained in the non-adv sgfs files
    '''

    def load_non_adv_model_data(self):
        
        directory = os.fsdecode(self.non_adv_dir_path)

        non_adv_data = []

        for sub_folder in os.listdir(directory):
            sub_folder_path = self.non_adv_dir_path +'/'+ os.fsdecode(sub_folder)
            if not os.fsdecode(sub_folder) in self.cursed_files:
                for file in os.listdir(sub_folder_path):
                    if not os.fsdecode(file) in self.cursed_files:
                        file_path = sub_folder_path + '/' + os.fsdecode(file)
                        game_list = SGFReader.read_file(SGFReader, file_path)
                        for game in game_list:
                            game_info = ['NA','NA',game['pla'], game['opp'], 0.0, 
                                         'NA', game['moves'], game['result'],file_path]
                            non_adv_data.append(game_info)

        return(non_adv_data)

    '''
    ---description---
    function() -> list(str)
    extracts the information about games in the adversarial files.

    ---output---
    adv_data: the info of each game in the adversarial sgfs file
    '''
    def load_adv_model_data(self):
        
        directory = os.fsencode(self.adv_dir_path)

        adv_data = []

        for subfolder in os.listdir(directory):
            if not os.fsdecode(subfolder) in self.cursed_files:
                sub_folder_path = self.adv_dir_path + '/' + os.fsdecode(subfolder)
                for file in os.listdir(sub_folder_path):
                    if not os.fsdecode(file) in self.cursed_files:
                        file_path = sub_folder_path + '/' + os.fsdecode(file)
                        games_list = SGFReader.read_file(SGFReader, file_path)
                        for game in games_list:

                            black_name = 'NA'
                            white_name = 'NA'

                            black_model = game['pla']
                            white_model = game['opp']

                            #Finds the model name
                            for name in self.name_mapping.keys():
                                if black_model in self.name_mapping[name]:
                                    black_name = name
                                elif white_model in self.name_mapping[name]:
                                    white_name = name

                            #Determines if a game is between adversarial policy and target model
                            adversarial = 1.0

                            if ([black_name, white_name] in self.non_adv_matchups) or ([white_name, black_name] in self.non_adv_matchups):
                                adversarial = 0.0
                            #Handles special case of cont v dec23, where adversarial is determined by no. visits
                            elif (black_name in ['cont','dec23']) and (white_name in ['cont','dec23']):
                                if (black_model in ['b18-s8527m-v4096', 'b18-s8527m-v8192']):
                                    adversarial = 0.0
                                elif(white_model in ['b18-s8527m-v4096', 'b18-s8527m-v8192']):
                                    adversarial = 0.0
                            
                            #Determines the adversarial attack type
                            if (black_name  == 'gift') or (white_name == 'gift'):
                                attack_type = 'non-cyclic'
                            else:
                                attack_type = 'cyclic'

                            game_info = [black_name, white_name, black_model, white_model, adversarial, 
                                         attack_type, game['moves'], game['result'], file_path]

                            adv_data.append(game_info)
        
        return(adv_data)
    
    '''
    ---description---
    Splits games into batches that can be probed indivdually.

    ---parametres---
    batch:int - the number of games in each batch

    ---ouput---
    batches:List - list of games grouped by batches
    '''

    def batch_data(self, batch_size : int):

        non_adv_data = self.load_non_adv_model_data()

        adv_data = self.load_adv_model_data()

        data = non_adv_data + adv_data

        random.shuffle(data) #Seed set at top of file

        batches = []

        data_size = len(data) -1 

        for i in range(0, data_size, batch_size):
            if i + batch_size < data_size:
                batches.append(data[i : i+batch_size])
            elif i != data_size:
                batches.append(data[i : data_size])

        return(batches)
    

    def equal_batch_data(self, batch_size : int):
                
        non_adv_data = self.load_non_adv_model_data()

        adv_data = self.load_adv_model_data()

        batches = []

        data_size = min(len(non_adv_data), len(adv_data)) - 1

        class_batch_size = int(batch_size/2)

        for i in range (0, data_size, class_batch_size):

            if i + batch_size < data_size:
                current_batch = non_adv_data[i : i+class_batch_size] + adv_data[i : i+class_batch_size]
            elif i != data_size:
                current_batch = non_adv_data[i : data_size] + adv_data[i : data_size]

            batches.append(current_batch)
        
        return(batches)


    '''
    ---description---
    Plays games through a KataGo model and stores the specified layer outputs in separate directories.
    Directory Structure:
    save_directory
    |---model_name
    |    |
    |    |---Layer-Name1
    |    |   |---game_0.npy
    |    |   |---game_1.npy
    |    |   |---...
    |    |
    |    |---Layer_Name2
    |        |---...
    |
    |---meta_data.json
        ...

    The method creates the directory structure itself.
    you only need the save_directory to exist in your file structure.

    ---parametres---
    chkpt_file - KataGo model chkpt file, this is the KataGo models which layer outputs are stored
    model_name: str - the model name used to create the above direcotry structure
    extra_outputs - The layer outputs you wanted to be stored. shoudl be in the form <layername>.out
    batched_data - The game files you want to probe, separated into batches. (the output od batch_data())
    batch_index:int - the index of the batch you want to probe 
    save_directory:str - the directory path where you want to save the probe data, see directory structure above.
    '''
    def probe_batch(self, chkpt_file : str, model_name : str, extra_outputs : list, batched_data : list,batch_index : int, save_directory : str):

        
        for output in extra_outputs:
            directory_name = save_directory + '/' + model_name + '/' + (output.split('.')[0])
            directory_path = Path(directory_name)
            try:
                directory_path.mkdir(parents=True, exist_ok=True)
                print(f"Directory '{directory_name}' created successfully.")
            except FileExistsError:
                print(print(f"Directory '{directory_name}' already exists. Writing outputs to '{directory_name}"))        
        
        kata_model, kata_swa_model, other_state_dict = load_model.load_model(chkpt_file, use_swa=True, device = torch.device("cpu"))

        meta_file = Path(save_directory + '/' + model_name + '/meta_data.json')


        if not meta_file.is_file():
            
            meta_data = {}

            model_data = kata_model.config

            game_data = []

            for batch in batched_data:
                for game in batch:
                    game_meta_data = game[0:5]
                    game_data.append(game_meta_data)

            meta_data['model data'] = model_data
            meta_data['game data'] = game_data

            with open(meta_file, 'w') as f:
                json.dump(meta_data, f)
                
        batch = batched_data[batch_index]

        for i in range(0, len(batch)):

            game_data = batch[i]

            index = (len(batched_data[0]) * batch_index) + i

            gs = GameState(19, GameState.RULES_TT)

            #black_name = game_data[0]
            #black_model = game_data[2]
            #white_name = game_data[1]
            #white_model = game_data[3]
            adversarial = game_data[4]
            #attack_type = game_data[5]
            moves = game_data[6]
            #result = game_data[7]

            probe_outputs = {key: [] for key in extra_outputs}

            for j in range(0, len(moves)):

                move = moves[j]

                SGFReader.play_move(SGFReader, gs, move)

                model_outputs = gs.get_model_outputs(model=kata_model, extra_output_names = extra_outputs)
                
                for name in extra_outputs:
                    probe_outputs[name].append(
                            {
                            'move num' : j, 
                            'layer activation' : model_outputs[name], 
                            'adversarial' : adversarial
                            })
                    

            for output in extra_outputs:
                directory_name = save_directory + '/' + model_name + '/' + (output.split('.')[0])
                file_name = 'game_' + str(index) + '.npy'
                out_file = directory_name + '/' + file_name
                output_array = np.array(probe_outputs[output])
                with open(out_file, 'wb') as f:
                    np.save(f, output_array, allow_pickle = True)






                
