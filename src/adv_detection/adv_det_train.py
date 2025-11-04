
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader
from adv_detection.adv_det_dataset import Adversarial_Detection_Dataset as adv_dataset
from adv_detection.adv_det_model import Model
import wandb
import random

'''
---description---
Training loop for adversarial detection model.
---input---
data_dir_path: file path to directory where data is stored
model = None: The model you owuld like to train If left as none creates model instance from metadata.
'''    
class Training_Loop():

    def __init__(self, data_dir_path : str, batch_size : int, num_workers : int, learing_rate, model = None):

        self.data_dir_path = data_dir_path

        self.batch_sz = batch_size

        self.lr = learing_rate

        data_dir = os.fsencode(data_dir_path)
        file_list = os.listdir(data_dir)
        random.seed(42)
        random.shuffle(file_list)
        
        train_size = int(0.8 * len(file_list))

        self.train_dataset = adv_dataset(data_dir_path, file_list[0 : train_size])
        self.val_dataset = adv_dataset(data_dir_path, file_list[train_size+1 : -1])

        meta_dir = os.path.dirname(self.data_dir_path)
        meta_data_path =  meta_dir +'/'+'meta_data.json'
        with open(meta_data_path, 'r') as f:
            self.meta_data = json.load(f)
            f.close()
        
        if model == None:
            kata_model_config = self.meta_data['model data']
            self.model = Model(kata_model_config, 19)
        else:
            self.model = model

        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_sz, shuffle=False, num_workers = num_workers)

        self.val_loader = DataLoader(self.val_dataset, batch_size = self.batch_sz, shuffle=False, num_workers = num_workers)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self. optimizer = torch.optim.Adam(self.model.parameters(), lr = learing_rate)

    '''
    ---description---
    Just trains the model.
    ---input---
    num_epochs: int: number of trainingn epochs to perform
    '''    
    def train(self, num_epochs : int):

        for i in range(num_epochs):

            self.train_epoch()

            save_path = '../../models/detection' + '/model_' + str(i) + '.chkpt'
            torch.save(self.model, save_path)


    '''
    ---description---
    Trains model and logs the training with weights and biases.
    ---input---
    num_epochs: int: the number of training epocjs to perform.
    wandb_key: str: your api key for accessing weights and biases.
    '''    
    def train_and_log(self, num_epochs : int, run_name : str, wandb_key : str):

        wandb.login(key = wandb_key)

        run = wandb.init(
            project = run_name,
            config={
                "learning rate" : self.lr,
                "epochs" : num_epochs,
                "batch size" : self.batch_sz,
            },
        )

        for i in range(num_epochs):

            print('epoch: ' + str(num_epochs))

            last_loss, training_loss, eval_loss, eval_accuracy = self.train_epoch()
            wandb.log({'training loss': last_loss, 'eval accuracy' : eval_accuracy, 'eval loss' : eval_loss})
            print(eval_accuracy)

            save_path = '../../models/detection' + '/model_' + str(i) + '.chkpt'
            torch.save(self.model, save_path)
        
        wandb.finish()

    '''
    ---description---
    Performs a single epoch of trainging.
    ---output---
    last_batch_loss: average loss from previous batch
    training_loss: loss on the training set
    eval_loss: loss on eval set 
    eval_accuracy: accuracy on the eval set
    '''    
    def train_epoch(self):

        running_loss = 0.
        average_batch_loss = 0.

        training_loss = []
        for i, data in enumerate(self.train_loader):

            indexes, inputs, labels = data

            labels = torch.tensor(labels)
            labels = labels.long()

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            outputs = outputs.float()
            
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            average_batch_loss = running_loss / self.batch_sz

            #print('  batch {} loss: {}'.format(i + 1, average_batch_loss))
            training_loss.append(average_batch_loss)
            running_loss = 0.
        
        last_batch_loss = average_batch_loss
        eval_loss, eval_accuracy = self.eval_epoch()
        print("Acc: " + str(eval_accuracy))

        return (last_batch_loss, training_loss, eval_loss, eval_accuracy)
    
    '''
    ---description---
    Evaluates current models performance.
    ---output---
    eval_loss: loss on the eval set 
    accurcy: accuracy on the evaluation set 
    '''    
    def eval_epoch(self):

        self.model.eval()

        correct = []
        running_loss = 0
        loss_fn = torch.nn.CrossEntropyLoss()
        val_size = 0

        with torch.no_grad():

            for i, data in enumerate(self.val_loader):

                indexes, inputs, labels = data

                val_size += len(labels)

                labels = torch.tensor(labels)
                labels = labels.long()

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                outputs = outputs.float()

                loss = loss_fn(outputs, labels)

                predicted = [0 if n[0] > n[1] else 1 for n in outputs]

                correct.append([1 if p == l else 0 for p,l in zip(predicted, labels)])

                running_loss += loss.item()
        
        eval_loss = running_loss/val_size

        num_correct = sum([sum(n) for n in correct])

        accuracy = float(num_correct)/float(val_size)

        return(eval_loss, accuracy)
    

    def evaluate_model(self):
            
            self.model.eval()
            
            num_correct = 0

            correct_per_move_num = {}
            correct_per_move_adv = {}
            correct_per_move_non_adv = {}
            correct_per_adv_model = {}

            adv_model_names = ['atari', 'base-adv', 'cont', 'gift', 'large', 'stall', 'ViT-adv']
            for model in adv_model_names:
                correct_per_adv_model[model] = []

            with torch.no_grad():

                for i, (indexes, inputs, labels) in enumerate(self.val_loader):

                    for j, (idx, input, label) in enumerate(zip(indexes, inputs, labels)):

                        label = label.long()

                        self.optimizer.zero_grad

                        output = self.model(input)[0]
                        output = output.float()

                        if output[0] > output[1]:
                            predicted = 0.0
                        else: 
                            predicted = 1.0

                        correct = int(predicted == label)

                        move_num = self.val_dataset.get_sample(idx)['move num']

                        if (move_num) in correct_per_move_num.keys():
                            correct_per_move_num[(move_num)].append(correct)
                        else:
                            correct_per_move_num[(move_num)] = [correct]

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
                                
                            
                            meta_data = self.val_dataset.get_meta_data_from_idx(idx)

                            model1 = meta_data[0]
                            model2 = meta_data[1]
                            adv = meta_data[-1]
                            if model1 in adv_model_names:
                                correct_per_adv_model[model1].append(correct)
                            elif model2 in adv_model_names:
                                correct_per_adv_model[model2].append(correct)
                            else:
                                if model1 != 'NA':
                                    print('error uknow model :' + model1)
                                if model2 != 'NA':
                                    print('error uknow model :' + model1)                

            return(correct_per_move_num, correct_per_move_adv, correct_per_move_non_adv, correct_per_adv_model)




    


