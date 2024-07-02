import os
import torch
from .dataset import AstroDataset


class prep_loaders:
    def __init__(self, config):
        self.config = config
        self.dpaths = config['dpaths']
        self.dhypar = config['dhypar']
        
        if self.dhypar['set_val']:
            self._keys = ['train_set', 'val_set', 'test_set']
        else:
            self._keys = ['train_set', 'test_set']

        self.make()

    def make(self):
        datasets = self._prep_datasets()

        loaders = {}
        for i in range(len(datasets)):
            if list(datasets.keys())[i] == 'train_set':
                shuffle = True
            else:
                shuffle = False
                
            loader = torch.utils.data.DataLoader(
                datasets[self._keys[i]],
                batch_size=int(self.dhypar['batch']),
                shuffle=shuffle, num_workers=1,
                drop_last=True
            )
            loaders[self._keys[i]] = loader
            
        self.loaders = loaders


    def _prep_datasets(self):
        if not self.dhypar['cross_valid']:
            if bool(self.dhypar['set_val']):
                paths = [
                    os.path.join(self.dpaths['label_folder'], self.dpaths['train_label']),
                    os.path.join(self.dpaths['label_folder'], self.dpaths['val_label']),
                    os.path.join(self.dpaths['label_folder'], self.dpaths['test_label'])
                ]
            else:
                paths = [
                    os.path.join(self.dpaths['label_folder'], self.dpaths['train_label']),
                    os.path.join(self.dpaths['label_folder'], self.dpaths['test_label'])
                ]
        else:    
            paths = prep_data_cv(
                self.dpaths['label_folder'],
                fold_num=int(self.dhypar['fold_num']),
                set_val=bool(self.dhypar['set_val']),
                test_num=int(self.dhypar['test_id']),
                combine_label_name=f'combined_train_{self.dpaths["output_version"]}.csv'
            )

        sets = {}
        for i in range(len(paths)):
            _set = AstroDataset(
                label_file=paths[i],
                mask_en=self.dhypar['energy_range'],
                class_idx=self.dhypar['class_tag'],
                setlog=bool(self.dhypar['set_log']),
                epsilon=float(self.dhypar['epsilon']),
                norm_method={
                    'method':self.dhypar['norm_method'],
                    'norm_const':float(self.dhypar['norm_const']),
                    'norm_min_max':self.dhypar['norm_min_max']
                }
            )
            sets[self._keys[i]] = _set

        return sets


def prep_data_cv(label_path,
                 fold_num=10,
                 test_num=0,
                 set_val=False,
                 combine_label_name='combined_train.csv'):
    if set_val:
        paths = _prep_data_val(
            label_path,
            fold_num=fold_num,
            test_num=test_num,
            combine_label_name=combine_label_name
        )
    else:
        paths = _prep_data_no_val(
            label_path,
            fold_num=fold_num,
            test_num=test_num,
            combine_label_name=combine_label_name
        )
    return paths


def _prep_data_no_val(label_path, fold_num=10, test_num=0,
                      combine_label_name='combined_train.csv'):
    test_label_path = os.path.join(
        label_path, str(test_num)+'.csv'
    )

    filenames = []
    for i in range(fold_num):
        filenames.append(
            os.path.join(label_path, '%d.csv'%i)
        )

    filenames.remove(test_label_path)

    with open(os.path.join(label_path, combine_label_name), 'w') as train_list:
        for fold in filenames:
            for line in open(fold, 'r'):
                train_list.write(line)
    train_label_path = os.path.join(
        label_path, combine_label_name
    )

    return train_label_path, test_label_path

def _prep_data_val(label_path, fold_num=10, test_num=0,
                   combine_label_name='combined_train.csv'):
    test_label_path = os.path.join(label_path, str(test_num)+'.csv')

    if test_num == 9:
        val_num = 0
    else:
        val_num = test_num + 1
    val_label_path = os.path.join(label_path, str(val_num)+'.csv')

    filenames = []
    for i in range(fold_num):
        filenames.append(os.path.join(label_path, '%d.csv'%i))

    filenames.remove(test_label_path)
    filenames.remove(val_label_path)

    with open(os.path.join(label_path, combine_label_name), 'w') as train_list:
        for fold in filenames:
            for line in open(fold, 'r'):
                train_list.write(line)
    train_label_path = os.path.join(label_path, combine_label_name)

    return train_label_path, val_label_path, test_label_path