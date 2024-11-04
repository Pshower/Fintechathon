import torch
import collections
import os
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed


ASVFile = collections.namedtuple('ASVFile',
    ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])

class ASVDataset(Dataset):
    """ Utility class to load  train/dev datatsets """
    def __init__(self, database_path=None,protocols_path=None,transform=None, 
        is_train=True, sample_size=None, 
        is_logical=True, feature_name=None, is_eval=False,
        eval_part=0):

        track = 'LA'   
        data_root=protocols_path      
        assert feature_name is not None, 'must provide feature name'
        self.track = track
        self.is_logical = is_logical
        self.prefix = 'ASVspoof2019_{}'.format(track)
        
        v1_suffix = ''
        if is_eval and track == 'LA':
            v1_suffix='_v1'
            self.sysid_dict = {
            '-': 0,  # bonafide speech
            'A07': 1, 
            'A08': 2, 
            'A09': 3, 
            'A10': 4, 
            'A11': 5, 
            'A12': 6,
            'A13': 7, 
            'A14': 8, 
            'A15': 9, 
            'A16': 10, 
            'A17': 11, 
            'A18': 12,
            'A19': 13,
        }
        else:
            self.sysid_dict = {
            '-': 0,  # bonafide speech
            
            'A01': 1, 
            'A02': 2, 
            'A03': 3, 
            'A04': 4, 
            'A05': 5, 
            'A06': 6,
             
          
        }

        self.data_root_dir=database_path   
        self.is_eval = is_eval
        self.sysid_dict_inv = {v:k for k,v in self.sysid_dict.items()}
        print('sysid_dict_inv',self.sysid_dict_inv)

        self.data_root = data_root
        print('data_root',self.data_root)

        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        print('dset_name',self.dset_name)

        self.protocols_fname = 'eval.trl' if is_eval else 'train.trn' if is_train else 'dev.trl'
        print('protocols_fname',self.protocols_fname)
        
        self.protocols_dir = os.path.join(self.data_root)
        print('protocols_dir',self.protocols_dir)
        
        self.files_dir = os.path.join(self.data_root_dir, '{}_{}'.format(
            self.prefix, self.dset_name ), 'flac')
        print('files_dir',self.files_dir)
        # /home/gjw/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_train/flac/ASVspoof2019_LA_train/flac
        # 多了一个, 所以data_root_dir 应该是 /home/gjw/Datasets/ASVSpoof2019/LA/
        self.protocols_fname = os.path.join(self.protocols_dir,
            'ASVspoof2019.{}.cm.{}.txt'.format(track, self.protocols_fname))
        
        print('protocols_file',self.protocols_fname)

        self.cache_fname = 'cache_{}_{}_{}.npy'.format(self.dset_name,track,feature_name)
        print('cache_fname',self.cache_fname)
        
        # data_root /home/gjw/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/
        # dset_name dev
        # protocols_fname dev.trl

        # protocols_dir /home/gjw/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/
        # files_dir /home/gjw/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_train/flac/ASVspoof2019_LA_dev/flac
        # protocols_file /home/gjw/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
        
        self.transform = transform

        if os.path.exists(self.cache_fname):
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname, weights_only=True)
            print('Dataset loaded from cache ', self.cache_fname)
        else:
            self.files_meta = self.parse_protocols_file(self.protocols_fname)
            ###
            data = list(map(self.read_file, self.files_meta))
            ###
            self.data_x, self.data_y, self.data_sysid = map(list, zip(*data))
            
            if self.transform:
                self.data_x = Parallel(n_jobs=4, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)
            torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
            
        if sample_size:
            select_idx = np.random.choice(len(self.files_meta), size=(sample_size,), replace=True).astype(np.int32)
            self.files_meta= [self.files_meta[x] for x in select_idx]
            self.data_x = [self.data_x[x] for x in select_idx]
            self.data_y = [self.data_y[x] for x in select_idx]
            self.data_sysid = [self.data_sysid[x] for x in select_idx]
            
        self.length = len(self.data_x)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        
        data_x, sample_rate = sf.read(meta.path)
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')
        if self.is_eval:
            return ASVFile(speaker_id=tokens[0],
                file_name=tokens[1],
                path=os.path.join(self.files_dir, tokens[1] + '.flac'),
                sys_id=self.sysid_dict[tokens[3]],
                key=int(tokens[4] == 'bonafide'))
        return ASVFile(speaker_id=tokens[0],
            file_name=tokens[1],
            path=os.path.join(self.files_dir, tokens[1] + '.flac'),
            sys_id=self.sysid_dict[tokens[3]],
            key=int(tokens[4] == 'bonafide'))
        

   
    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)

   


if __name__ == '__main__':
    from torchvision import transforms
    from torch import Tensor
    import soundfile as sf

    # 读取FLAC文件
    data, samplerate = sf.read('/home/gjw/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_train/flac/LA_T_9999995.flac')
    # data变量包含了音频数据，samplerate是采样率
    print(f"采样率: {samplerate}")
    print(f"音频数据: {data}")


    def pad(x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        # need to pad
        num_repeats = int(max_len / x_len)+1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x 
    
    database_path = '/home/gjw/Datasets/ASVSpoof2019/LA/'
    protocols_path = '/home/gjw/Datasets/ASVSpoof2019/LA/ASVspoof2019_LA_cm_protocols/'
    is_logical = True
    transforms = transforms.Compose([
        lambda x: pad(x),
        lambda x: Tensor(x)
    ])
    features = 'Raw_audio'
    train_set = ASVDataset(database_path=database_path,protocols_path=protocols_path,is_train=True, is_logical=is_logical, transform=transforms,
                                      feature_name=features)
    print("load success!")
    

  

