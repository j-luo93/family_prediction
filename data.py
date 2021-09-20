from typing import Dict, List, Tuple
import re
import panphon as ph
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import BATCH_SIZE


class OneLangDataset(Dataset):
    """Dataset for one language."""

    def __init__(self, np_arr, labels):
        self.data = torch.from_numpy(np_arr).float()
        self.labels = torch.from_numpy(labels).long()
        assert len(self.data) == len(self.labels)

    def __getitem__(self, idx: int):
        return {
            'data': self.data[idx],
            'label': self.labels[idx]
        }

    def __len__(self):
        return len(self.data)


class FamilyDataset(Dataset):
    """Dataset for one family."""

    def __init__(self, lang2dataset: Dict[str, OneLangDataset], langind):
        langs = sorted(lang2dataset)
        self.data = torch.cat([lang2dataset[lang].data for lang in langs], dim=0)
        self.labels = torch.LongTensor(np.full((len(self.data),), langind))

        self.langs = list()
        for lang in langs:
            self.langs.extend([lang] * len(lang2dataset[lang]))

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'label': self.labels[idx],
            'lang': self.langs[idx]
        }

    def __len__(self):
        return len(self.data)


class CombinedDataset(Dataset):

    def __init__(self, fam2dataset: Dict[str, FamilyDataset]):
        data = list()
        labels = list()
        langs = list()
        for fam, dataset in fam2dataset.items():
            data.append(dataset.data)
            labels.append(dataset.labels)
            langs.extend(dataset.langs)
        self.data = torch.cat(data, dim=0)
        self.labels = torch.cat(labels, dim=0)
        self.langs = langs

    def __getitem__(self, idx: int):
        return {
            'data': self.data[idx],
            'label': self.labels[idx],
            'lang': self.langs[idx]
        }

    def __len__(self):
        return len(self.data)


def get_dataset() -> Tuple[List[str], CombinedDataset]:
    removelist = ['livv1243', 'lule1254', 'inar1241', 'bela1254',
                  'latv1249', 'lati1261', 'kaza1248', 'buri1258', 'darg1241']

    words = pd.read_csv('northeuralex-0.9-forms.tsv',
                        sep='\t')
    langs = pd.read_csv('northeuralex-0.9-language-data.tsv',
                        sep='\t')

    wordindices = []
    for index, i in enumerate(words['Glottocode']):
        if i not in removelist:
            wordindices.append(index)
    words = words.iloc[wordindices]
    wordindices = []
    for index, i in enumerate(langs['glotto_code']):
        if i not in removelist:
            wordindices.append(index)
    langs = langs.iloc[wordindices]

    words = words[pd.notnull(words['rawIPA'])]

    gltcode = langs['glotto_code'].tolist()
    subfam = langs['subfamily'].tolist()
    lnames = langs['name'].tolist()

    # import collections
    # counter=collections.Counter(langs['subfamily'])
    # print(counter)

    idx = [index for index, element in enumerate(subfam) if element in ['Italic', 'Germanic', 'Balto-Slavic', 'Finnic', 'Saami', 'Daghestanian', 'Samoyedic',
                                                                        'Iranian', 'South Dravidian'  # , 'Permian', 'Celtic', 'Kipchak', 'Eastern Mongolic'
                                                                        ]]
    codes = [gltcode[i] for i in idx]
    words2 = words[words['Glottocode'].isin(codes)]
    gltcodes_words = words['Glottocode'][words['Glottocode'].isin(codes)].tolist()

    family = []
    family_id = []
    """
    for i in range(len(words2)):
        fm = langs['subfamily'][langs['glotto_code'] == words2['Glottocode'].iloc[i]].tolist()[0]
        if fm == 'Italic':
            fmid = 0
        if fm == 'Germanic':
            fmid = 1
        if fm == 'Balto-Slavic':
            fmid = 2
        if fm == 'Finnic':
            fmid = 3
        if fm == 'Saami':
            fmid = 4
        if fm == 'Daghestanian':
            fmid = 5
        if fm == 'Samoyedic':
            fmid = 6
        if fm == 'Iranian':
            fmid = 7
        if fm == 'South Dravidian':
            fmid = 8
        if fm == 'Permian':
            fmid = 9
        if fm == 'Celtic':
            fmid = 10
        if fm == 'Kipchak':
            fmid = 11
        if fm == 'Eastern Mongolic':
            fmid = 12

        family.append(fm)
        family_id.append(fmid)
    """

    for i in range(len(words2)):
        fm = langs['subfamily'][langs['glotto_code'] == words2['Glottocode'].iloc[i]].tolist()[0]
        if fm == 'Italic':
            fmid = 0
        if fm == 'Germanic':
            fmid = 1
        if fm == 'Balto-Slavic':
            fmid = 2
        if fm == 'Finnic':
            fmid = 3
        if fm == 'Saami':
            fmid = 4
        if fm == 'Daghestanian':
            fmid = 5
        if fm == 'Samoyedic':
            fmid = 6
        if fm == 'Iranian':
            fmid = 7
        if fm == 'South Dravidian':
            fmid = 8

        family.append(fm)
        family_id.append(fmid)

    # wordlist = words2['rawIPA'].tolist()
    wordlist = words2['IPA'].tolist()
    langcodes = words2['Language_ID'].tolist()
    maxlen = 0
    for i in wordlist:
        if len(i) > maxlen:
            maxlen = len(i)
    """
    import pandas as pd
    phoible_df = pd.read_csv('C:/Users/Hartmann/Documents/Wissenschaft/Complexity/phoible.csv')
    #list(phoible_df)
    #p_uni = np.unique(phoible_df['LanguageName'])

    #len(list(set(lnames) & set(p_uni)))
    #p_gltcd = np.unique(phoible_df['Glottocode'].tolist())

    #len(list(set(gltcode) & set(p_gltcd)))
    failtable = []
    word_container = []
    for index, entry in enumerate(wordlist):
        print(index/len(wordlist))
        code = gltcodes_words[index]
        subdf = phoible_df[phoible_df['Glottocode'] == code]
        wordarray = []
        new_entry = entry.split(' ')
        for sound in new_entry:
            try:
                wordarray.append(np.asarray(subdf.iloc[:,11:48][subdf['Phoneme'] == sound].iloc[0,:]))
            except:
                try:
                    wordarray.append(np.asarray(phoible_df.iloc[:,11:48][phoible_df['Phoneme'] == sound].iloc[0,:]))
                except:
                    failtable.append(sound)
                    wordarray.append(np.zeros((1,37)))

        w = np.vstack(wordarray)
        w = np.transpose(w)
        w = np.where(w == "-", 0, w)
        w = np.where(w == "+, -", 1, w)
        w = np.where(w=="0", 0.5, w)
        w = np.where(w=="+", 1, w)
        while np.shape(w)[1] <maxlen:
            w = np.append(w, np.zeros((37,1)), axis=1)
        word_container.append(w)

    print(np.unique(failtable))

    np.save('phoibledata.npy', word_container)
    """

    word_container = np.load('phoibledata.npy', allow_pickle=True)

    for index, i in enumerate(word_container):
        for endex, e in enumerate(i):
            for yndex, y in enumerate(e):
                if type(y) is str:
                    word_container[index][endex][yndex] = 1.0

    """
    """

    ft = ph.FeatureTable()

    word_container2 = []
    for entry in wordlist:
        entry = re.sub('͡', '', entry)
        entry = re.sub('kʼʷ', 'kʷʼ', entry)
        entry = re.sub('qʰʷ', 'qʷʰ', entry)
        entry = re.sub('qʼʷ', 'qʷʼ', entry)
        entry = re.sub('qʼˤ', 'qˤʼ', entry)
        entry = re.sub('tʰʲ', 'tʲʰ', entry)
        entry = re.sub('tʼʷ', 'tʷʼ', entry)
        entry = re.sub('tʼʷ', 'tʷʼ', entry)
        entry = re.sub(' ʰ', 'ʰ', entry)
        entry = re.sub(' ʼ', 'ʼ', entry)
        entry = re.sub(' ˀ', 'ˀ', entry)
        entry = re.sub(' ˤ', 'ˤ', entry)
        w = ft.word_array(
            ["syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid", "voi", "sg", "cg", "ant", "cor", "distr",
             "lab", "hi", "lo", "back", "round", "velaric", "tense", "long"], entry)
        w = np.transpose(w)
        w = np.where(w == 0, 0.5, w)
        w = np.where(w == -1, 0, w)
        while np.shape(w)[1] < maxlen:
            w = np.append(w, np.zeros((22, 1)), axis=1)
        word_container2.append(w)
    #word_container2 = word_container2[:-1]

    stacked = np.hstack((word_container, np.asarray(word_container2)))
    word_container = stacked
    del word_container2

    firstsum = np.sum(word_container, axis=1)
    secondsum = np.sum(firstsum, axis=0)

    word_container = word_container[:, :, :secondsum.tolist().index(0.0)]

    word_container_new = []
    for i in word_container:
        newarray = np.zeros((59 * 3, 25))
        for col in range(25):
            for row in range(59):
                if i[row, col] == 0.5:
                    newarray[(row * 3):((row * 3) + 3), col][1] = 1
                elif i[row, col] == 1:
                    newarray[(row * 3):((row * 3) + 3), col][2] = 1
                else:
                    newarray[(row * 3):((row * 3) + 3), col][0] = 1
        word_container_new.append(newarray)

    num_words = 10
    langcodes_uni = np.unique(langcodes)
    cubes = []
    cube_labels = []
    cube_lang_labels = []
    for i in langcodes_uni:
        indxs = [index for index, value in enumerate(langcodes) if value == i]
        tmplist = []
        counter = 0
        for e in indxs:
            if counter < num_words:
                try:
                    tmplist.append(word_container_new[e])
                except:
                    continue
                counter += 1
            else:
                tmplist = np.stack(tmplist)
                cubes.append(tmplist)
                cube_lang_labels.append(i)
                cube_labels.append(family_id[e])
                tmplist = []
                counter = 0

    famdict_ita = {}
    famdict_germ = {}
    famdict_Finnic = {}
    famdict_BaltoSlavic = {}
    famdict_Saami = {}
    famdict_Daghestanian = {}
    famdict_Samoyedic = {}
    famdict_Iranian = {}
    famdict_SouthDravidian = {}
    famdict_Permian = {}
    famdict_Celtic = {}
    famdict_Kipchak = {}
    famdict_EasternMongolic = {}
    langdict_indiv = {}

    for i in langcodes_uni:
        indxs = [index for index, value in enumerate(cube_lang_labels) if value == i]
        globals()[i + '_dataset'] = OneLangDataset(np.asarray([cubes[t] for t in indxs]).astype('float64'),
                                                   np.asarray([cube_labels[t] for t in indxs]))
        if i in ['dan', 'deu', 'eng', 'isl', 'nld', 'nor', 'swe']:
            famdict_germ[i] = globals()[i + '_dataset']
        elif i in ['cat', 'fra', 'ita', 'lat', 'por', 'ron', 'spa']:
            famdict_ita[i] = globals()[i + '_dataset']
        elif i in ['ekk', 'fin', 'krl', 'liv', 'olo', 'vep']:
            famdict_Finnic[i] = globals()[i + '_dataset']
        elif i in ['bel', 'bul', 'ces', 'hrv', 'lav', 'lit', 'pol', 'rus', 'slk',
                   'slv', 'ukr']:
            famdict_BaltoSlavic[i] = globals()[i + '_dataset']
        elif i in ['sjd', 'sma', 'sme', 'smj', 'smn', 'sms']:
            famdict_Saami[i] = globals()[i + '_dataset']
        elif i in ['ava', 'dar', 'ddo', 'lbe', 'lez']:
            famdict_Daghestanian[i] = globals()[i + '_dataset']
        elif i in ['enf', 'nio', 'sel', 'yrk']:
            famdict_Samoyedic[i] = globals()[i + '_dataset']
        elif i in ['kmr', 'oss', 'pbu', 'pes']:
            famdict_Iranian[i] = globals()[i + '_dataset']
        elif i in ['kan', 'mal', 'tam', 'tel']:
            famdict_SouthDravidian[i] = globals()[i + '_dataset']
        # elif i in ['koi', 'kpv', 'udm']:
        #    famdict_Permian[i] = globals()[i + '_dataset']
        # elif i in ['bre', 'cym', 'gle']:
        #    famdict_Celtic[i] = globals()[i + '_dataset']
        # elif i in ['bak', 'kaz', 'tat']:
        #    famdict_Kipchak[i] = globals()[i + '_dataset']
        # elif i in ['bua', 'khk', 'xal']:
        #    famdict_EasternMongolic[i] = globals()[i + '_dataset']
        langdict_indiv[i] = DataLoader(globals()[i + '_dataset'], shuffle=True, batch_size=BATCH_SIZE, drop_last=True)

    # fr_dataset = OneLangDataset(train_data[:1000], train_labels[:1000])
    # pt_dataset = OneLangDataset(train_data[1000:2000], train_labels[1000:2000])

    # rom_dataset = FamilyDataset({'fr': fr_dataset, 'pt': pt_dataset})
    rom_dataset = FamilyDataset(famdict_ita, 0)
    germ_dataset = FamilyDataset(famdict_germ, 1)
    BaltoSlavic_dataset = FamilyDataset(famdict_BaltoSlavic, 2)
    Finnic_dataset = FamilyDataset(famdict_Finnic, 3)
    Saami_dataset = FamilyDataset(famdict_Saami, 4)
    Daghestanian_dataset = FamilyDataset(famdict_Daghestanian, 5)
    Samoyedic_dataset = FamilyDataset(famdict_Samoyedic, 6)
    Iranian_dataset = FamilyDataset(famdict_Iranian, 7)
    SouthDravidian_dataset = FamilyDataset(famdict_SouthDravidian, 8)
    #Permian_dataset = FamilyDataset(famdict_Permian, 9)
    #Celtic_dataset = FamilyDataset(famdict_Celtic, 10)
    #Kipchak_dataset = FamilyDataset(famdict_Kipchak, 11)
    #EasternMongolic_dataset = FamilyDataset(famdict_EasternMongolic, 12)

    # langdict_indiv = {}

    # comb = []
    # for i in rom_dataset:
    #     comb.append(i)
    # for i in germ_dataset:
    #     comb.append(i)

    # import random
    # random.shuffle(comb)

    # comb_dl = DataLoader(comb, shuffle=True, batch_size=batch_size, drop_last=True)
    comb_dataset = CombinedDataset({'rom': rom_dataset, 'germ': germ_dataset, 'Finnic': Finnic_dataset,
                                    'BaltoSlavic': BaltoSlavic_dataset, 'Saami': Saami_dataset, 'Daghestanian': Daghestanian_dataset,
                                    'Samoyedic': Samoyedic_dataset, 'Iranian': Iranian_dataset, 'SouthDravidian': SouthDravidian_dataset  # ,
                                    # 'Permian': Permian_dataset, 'Celtic': Celtic_dataset, 'Kipchak': Kipchak_dataset, 'EasternMongolic': EasternMongolic_dataset

                                    })  # creates basically a huge list of dictionaries where each item is a dictionary
    langlist = list(langdict_indiv.keys())

    return langlist, comb_dataset

    # training_dataset = comb_dataset
