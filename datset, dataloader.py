import torch.utils.data

# 기본적으로 아래 두가지를 사용한다.
# torch.utils.data.DataLoader
# torch.utils.data.Dataset

"""  

Dataset => 샘플과 정답(label)을 저장한다.
DataLoader => 샘플에 쉽게 접근 가능하게 순회 가능한 객체 iterable로 감싼다.
DataLoader(Dateset) 과 같은 형태가 될까?

"""