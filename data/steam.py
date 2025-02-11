import numpy as np
import pandas as pd
import pickle
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple


class SteambaseDataset(Dataset):
    """
    Steam 기반 데이터셋을 로드하는 PyTorch Dataset 클래스
    """

    def __init__(
        self, 
        dataset_name: str, 
        df: pd.DataFrame, 
        description: List[Tuple[str, int, str]], 
        device: torch.device
    ) -> None:
        """
        데이터셋을 초기화하는 함수
        
        :param dataset_name: 데이터셋의 이름
        :param df: 데이터셋이 포함된 DataFrame
        :param description: 컬럼 설명 목록 (컬럼명, 크기, 타입)
        :param device: 데이터를 저장할 장치 (CPU 또는 GPU)
        """
        super(SteambaseDataset, self).__init__()
        self.dataset_name = dataset_name
        self.df = df
        self.length = len(df)

        # DataFrame의 각 컬럼을 PyTorch 텐서로 변환하여 저장
        self.name2array: Dict[str, torch.Tensor] = {
            name: torch.from_numpy(np.array(list(df[name])).reshape(self.length, -1)).to(device)
            for name in df.columns
        }

        # 컬럼 설명을 기반으로 데이터 형식 지정
        self.format(description)

        # 'interaction' 컬럼을 제외한 나머지를 입력(feature)로 사용
        self.features: List[str] = [name for name in df.columns if name != 'interaction']
        self.label: str = 'interaction'

    def format(self, description: List[Tuple[str, int, str]]) -> None:
        """
        컬럼 설명을 기반으로 데이터 타입을 변환하는 함수

        :param description: 컬럼 설명 목록 (컬럼명, 크기, 타입)
        """
        for name, _, type_ in description:  # size는 사용되지 않으므로 _로 변경
            if type_ in ('spr', 'seq'):  # 정수형 데이터 처리
                self.name2array[name] = self.name2array[name].to(torch.long)
            elif type_ == 'ctn':  # 연속형(실수형) 데이터 처리
                self.name2array[name] = self.name2array[name].to(torch.float32)
            elif type_ == 'label':  # 레이블은 변환하지 않음
                pass
            elif type_ == 'emb':
                self.name2array[name] = self.name2array[name].to(torch.float32)
            else:
                raise ValueError(f'알 수 없는 데이터 타입: {type_}')
                
    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        주어진 인덱스의 데이터 샘플을 반환하는 함수
        
        :param index: 가져올 데이터의 인덱스
        :return: (입력 데이터(features), 정답(label))
        """
        return (
            {name: self.name2array[name][index] for name in self.features}, 
            self.name2array[self.label][index].squeeze()
        )

    def __len__(self) -> int:
        """
        데이터셋의 전체 길이(샘플 개수)를 반환하는 함수
        """
        return self.length


class SteamColdStartDataLoader:
    """
    Steam 기반 데이터셋을 Cold Start 환경에서 로드하는 데이터로더 클래스
    """

    def __init__(
        self, 
        dataset_name: str, 
        dataset_path: str, 
        device: torch.device, 
        bsz: int = 32, 
        shuffle: bool = True
    ) -> None:
        """
        데이터셋을 로드하여 DataLoader 객체로 변환하는 함수
        
        :param dataset_name: 데이터셋의 이름
        :param dataset_path: 데이터셋 파일(.pkl)의 경로
        :param device: 데이터를 저장할 장치 (CPU 또는 GPU)
        :param bsz: 배치 크기 (기본값: 32)
        :param shuffle: 데이터 섞기 여부 (기본값: True)
        """
        assert os.path.exists(dataset_path), f'{dataset_path} 경로가 존재하지 않습니다.'

        # 데이터 파일을 열어서 로드
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)

        self.dataset_name: str = dataset_name
        self.dataloaders: Dict[str, DataLoader] = {}
        self.description: List[Tuple[str, int, str]] = data['description']

        # 데이터셋을 DataLoader로 변환
        for key, df in data.items():
            if key == 'description':  # 설명 정보는 건너뜀
                continue

            # DataLoader 생성 (metaE 포함 여부에 따라 shuffle 조정)
            self.dataloaders[key] = DataLoader(
                SteambaseDataset(dataset_name, df, self.description, device),
                batch_size=bsz,
                shuffle=False if 'metaE' in key else shuffle
            )

        # 데이터셋의 키 목록 저장
        self.keys: List[str] = list(self.dataloaders.keys())

        # 아이템 관련 특징 목록 정의
        self.item_features: List[str] = ['item_id', 'count', 'date_release_unix', 'price_original', 'required_age', 'tag_embedding', 'popular_developers']

    def __getitem__(self, name: str) -> DataLoader:
        """
        데이터셋에서 특정 이름의 DataLoader를 반환하는 함수
        
        :param name: 원하는 데이터셋의 이름
        :return: 해당 데이터셋의 DataLoader 객체
        """
        assert name in self.keys, f'{name} 데이터셋이 존재하지 않습니다.'
        return self.dataloaders[name]
