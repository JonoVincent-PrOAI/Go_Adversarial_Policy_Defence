from torch.utils.data import Dataset
import os
import numpy as np
import json
import sys

class Adversarial_Detection_Dataset(Dataset):
    '''
    ---description---
    Implements PyTorch Datatset. 
    --parametres---
    dir_path: str: path to where the probe data is stored. Each Game should be stored in a separate .npy file
    file_list (OPTIONAL): Should be output of os.listdir(). If a file_list is passed only the files in file_list will be used.
    '''

    def __init__(self, dir_path : str, file_list = None):

        self.dir_path = dir_path
        if file_list == None:
            data_dir = os.fsencode(dir_path)
            self.file_list = os.listdir(data_dir)
        else:
            self.file_list = file_list
        
        self.game_length_index = []
        for game_file in self.file_list:
            game_file = dir_path + '/' + os.fsdecode(game_file)
            with open(os.fsdecode(game_file), 'rb') as f:
                game_data = np.load(game_file, allow_pickle=True)
                self.game_length_index.append(len(game_data))

    def __len__(self):
        return(sum(self.game_length_index))

    '''
    ---description---
    Returns item at index idx.
    ---input---
    idx: int: the index of the dataset item you want to return
    ---output---
    idx: int: the index of returned item
    sample['layer activation']: pytorch.tensor: input to adv_det_model  
    sample['adversarial'] float: Binary value. Sample Label
    '''
    def __getitem__(self, idx : int):

        game_index = self.get_game_index_from_idx(idx)
        if game_index != None:
            move_index = idx - sum(self.game_length_index[: game_index])
            game_file = self.file_list[game_index]
            game_file_path = self.dir_path + '/' + os.fsdecode(game_file)
            with open(game_file_path, 'rb') as f:
                game_data = np.load(f, allow_pickle=True)
                sample = game_data[move_index]
                sample['game name'] = os.fsdecode(game_file)
                return(idx, sample['layer activation'], sample['adversarial'])

    '''
    ---description---
    Utility to retirve item with metadata.
    ---input---
    idx: int: index of item you want to retrieve
    ---output---
    sample: dict: Dict containing;
        'move num': point in the game where sample occured
        'game name': name of the file the sample is in
        'layer activation': pytorch tensor, input to adv_det_model
        'adversarial': binary value for if the move is from an adv game
    '''
    def get_item_and_info(self, idx : int):

        game_index = self.get_game_index_from_idx(idx)
        if game_index != None:
            move_index = idx - sum(self.game_length_index[: game_index])
            game_file = self.file_list[game_index]
            game_file_path = self.dir_path + '/' + os.fsdecode(game_file)
            with open(game_file_path, 'rb') as f:
                game_data = np.load(f, allow_pickle=True)
                sample = game_data[move_index]
                sample['game name'] = os.fsdecode(game_file)
                return(sample)
    '''
    ---description---
    Given an idx, returns the file_list index of the game that item is from.
    ---input---
    idx: int: idx of item, you want the game index for
    ---output---
    game: index: file_list index of game, will return None if idx is out of range of the dataset
    '''        
    def get_game_index_from_idx(self, idx : int):

        game_count = 0
        game_index = -1

        while idx >= game_count:
            game_index = game_index + 1
            game_count = game_count + self.game_length_index[game_index]
        
        if game_index > len(self.game_length_index):
            print('Error: Index' + str(idx) + ' out of range for dataset of size ' + 
                  str(sum(self.game_length_index)))
            return(None)
            
        else:
            return(game_index)
        
    '''
    ---description---
    Returns the game meta data of the item at index idx.
    ---input---
    idx: int: idx of item you want metadata on
    ---output---
    game_meta_data: dict: 
        'model data': dict: model config data for the katago model activations are taken from.
        'game data': list: model name sand versions for each game.
    '''    
    def get_meta_data_from_idx(self, idx : int):

        sample = self.get_item_and_info(idx)
        game_name = sample['game name']
        game_num = int(game_name.split('_')[1].replace('.npy', ''))

        meta_data_path = os.path.dirname(self.dir_path) + '/meta_data.json'
        meta_path = os.fsencode(meta_data_path)

        with open(meta_path) as meta_data:
            meta_file = json.load(meta_data)
            game_meta_data = meta_file['game data'][game_num]

            return(game_meta_data)
