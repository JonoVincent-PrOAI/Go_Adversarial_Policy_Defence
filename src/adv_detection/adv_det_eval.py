
import numpy as np
import os
import json
from torch.utils.data import DataLoader
from adv_detection.adv_det_dataset import Adversarial_Detection_Dataset as adv_dataset
from adv_detection.adv_det_model import Model
import sys

if '../../' not in sys.path:
    sys.path.append('../../')

'''
---description---
Class for running evaualtions of adv-det models
---input---
model: pytorch model: the model to evaluate
data_path: file path to eval set
batch_sz: batch size for data loader
num workers: number of workers for dataloader
'''
class Evaluation():

    def __init__(self, model, data_path, batch_sz, num_workers):

        self.batch_sz = batch_sz

        self.dataset = adv_dataset(data_path)
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_sz, shuffle=False, num_workers = num_workers)

        meta_data_path = os.path.dirname(data_path) + '/meta_data.json'
        self.meta_path = os.fsencode(meta_data_path)

        self.model = model

        self.eval_results = {}

        self.eval_results['input'] = []
        self.eval_results['label'] = []
        self.eval_results['predicted'] = []
        self.eval_results['game file'] = []
        self.eval_results['move num'] = []
    
    '''
    ---description---
    Runs the evaluation and stores results in self.eval_reults. Then returns the accuracy.
    ---output---
    the models accuracy
    '''
    def evaluate_model(self):

        num_correct = 0
        total = 0
        self.model.eval()

        for i, (indexes, inputs, labels) in enumerate(self.dataloader):

            for j, (idx, input, label) in enumerate(zip(indexes, inputs, labels)):

                label = label.long()

                output = self.model(input)[0]
                output = output.float()

                if output[0] > output[1]:
                    predicted = 0.0
                else: 
                    predicted = 1.0

                num_correct += int(predicted ==label)
                total += 1

                sample = self.dataset.get_item_and_info(idx)

                self.eval_results['input'].append(sample['layer activation'])
                self.eval_results['label'].append(sample['adversarial'])
                self.eval_results['predicted'].append(predicted)
                self.eval_results['game file'].append(sample['game name'])
                self.eval_results['move num'].append(sample['move num'])
        
        return(num_correct/total)
    
    '''
    ---description---
    Calculates the models accuracy on each adversarial policy in the dataset individually.
    ---output---
    dict: keys are model names, values are the accuracy on games inlcuding that model
    '''
    def evaluate_per_policy(self):

        adv_model_names = ['a9', 'atari', 'base-adv', 'cont', 'gift', 'large', 'stall', 'ViT-adv']

        correct_per_adv_model = dict.fromkeys(adv_model_names, 0)

        with open(self.meta_path) as meta_data:
            meta_file = json.load(meta_data)

            eval = zip(self.eval_results['label'], self.eval_results['predicted'], self.eval_results['game file'])
        
            for label,pred,game_name in eval:
                    
                game_num = int(game_name.split('_')[1].replace('.npy', ''))
                game_meta_data = meta_file['game data'][game_num]
                white_model_name = game_meta_data[0]
                black_model_name = game_meta_data[1]

                if white_model_name in adv_model_names:
                    correct_per_adv_model[white_model_name] += int(label == pred)
                elif black_model_name in adv_model_names:
                    correct_per_adv_model[black_model_name] += int(label == pred)
        
        return(correct_per_adv_model)

    '''
    ---description---
    Calculates teh accuracy of the model at each move number in teh game. Breaks this dwon to be per class
    ---output---
    three dict, keys are move numbers, values are accruacy at given move number
    '''
    def evaluate_per_move(self):

        correct_per_move = {}
        correct_per_move_adv = {}
        correct_per_move_non_adv = {}
        
        eval = zip(self.eval_results['label'], self.eval_results['predicted'], self.eval_results['move num'])

        for label, pred, move_num in eval:
            
            correct = int(pred == label)

            if (move_num) in correct_per_move.keys():
                correct_per_move[(move_num)].append(correct)
            else:
                correct_per_move[(move_num)] = [correct]

            if label == 0.0:
                if (move_num) in correct_per_move_non_adv.keys():
                    correct_per_move_non_adv[(move_num)].append(correct)
                else:
                    correct_per_move_non_adv[(move_num)] = [correct]
            else:
                if (move_num) in correct_per_move_adv.keys():
                    correct_per_move_adv[(move_num)].append(correct)
                else:
                    correct_per_move_adv[(move_num)] = [correct]
        
        for key in correct_per_move.keys():
            correct_per_move[key] = sum(correct_per_move[key]) / len(correct_per_move[key])

        for key in correct_per_move_adv.keys():
            correct_per_move_adv[key] = sum(correct_per_move_adv[key]) / len(correct_per_move_adv[key])
        
        for key in correct_per_move_non_adv.keys():
            correct_per_move_non_adv[key] = sum(correct_per_move_non_adv[key]) / len(correct_per_move_non_adv[key])

        return(correct_per_move, correct_per_move_adv, correct_per_move_non_adv)
    
    '''
    ---description---
    gives per class accuracy
    ---output---
    list, index is class labels, value are accuracy on class
    '''
    def evaluate_per_class(self):

        correct_per_class = [[], []]

        eval = zip(self.eval_results['label'], self.eval_results['predicted'])

        for label,pred in eval:

            correct_per_class[int(label)].append(int(label == pred))
        
        return([sum(correct_per_class[0]) / len(correct_per_class[0]), sum(correct_per_class[1]) / len(correct_per_class[1])])


