import numpy as np
import torch

# list 로 부터 tensor 생성하기
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

print(data)
print(x_data)
print()

# numpy array 로 텐서 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(np_array)
print(x_np)
print()

# numpy array 로 텐서 생성하기
x_ones = torch.ones_like(x_data)
x_twos = torch.full_like(x_data, 2)
x_rands = torch.rand_like(x_data, dtype=torch.float)

print(x_ones) # 텐서의 값이 1로 고정된다. ( 모양, 자료형만 유지 )
print(x_twos) # 텐서의 값이 2로 고정된다. ( 모양, 자료형만 유지 )
print(x_rands) # 텐서의 값이 랜덤으로 생성된다.
print()

# shape ( 모양 ) 으로 텐서 만들기
shape = (2,3,1) # 2차원 텐서 3 x 1
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

# tensor 모양 타입 장치 확인
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# gpu 사용가능한지 확인
if torch.cuda.is_available():
    print("available to gpu !")
    tensor = tensor.to("cuda")

# mpu 사용가능한지 확인 for. m1
if torch.backends.mps.is_built():
    print("available to mps")

# tensor 행,렬 값 체크 및 변경
tensor = torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,3] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print(y1)
print(y2)
print(y3)

# 텐서의 총 합 구하기 ( 추론의 예측률 뽑을때 많이 썻던거 같음 )
agg = tensor.sum().item()
print(agg)

# tensor 와 numpy 는 cpu 상에서 메모리를 공유한다. 따라서 하나를 변경하면 다른 하나도 변경됨
t = torch.ones(1)
n = t.numpy()
print(t, n)

t.add_(2)
print(t, n)


n = np.ones(5, dtype=int)
print(n)
t = torch.from_numpy(n)
print(t)
np.add(n, 1, out = n)
print(n, t)