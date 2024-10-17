'''
1. 顯示pytorch版本
'''
import torch 
print(torch.__version__) # 顯示目前安裝版本


'''
2. 檢查GPU與CUDA Toolkit是否存在
CUDA Toolkit: NVIDIA 開發的一套軟件開發工具套件。
專門用在GPU開發和運行並行計算應用程序。
'''
print(torch.cuda.is_available()) # True為檢測到GPU


'''
3. 使用toexh.tensor建立張量變數
 (1) pytorch會根據變數值決定資料型態
 (2) 也可以自己宣告型態: 
    1. 整數(torch.IntTensor)
    2. 長整數(torch.LongTensor)
    3. 浮點數(torch.FloatTensor)
'''
tensor = torch.tensor([
    [1, 2]
]) #(1,2)張量
print(tensor) 

tensor2 = torch.IntTensor([
    [1, 2]
]) 
print(tensor2) # 整數(torch.IntTensor)

tensor3 = torch.LongTensor([
    [1, 2]
]) 
print(tensor3) # 長整數(torch.LongTensor)

tensor4 = torch.FloatTensor([
    [1, 2]
])
print(tensor4) # 浮點數(torch.FloatTensor)

'''
4. 四則運算
'''
a = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
]) # (2,3)張量

b = torch.tensor([
    [9, 8, 7],
    [7, 6, 5]
]) # (2,3)張量
print('a+b:', a+b)
print('a-b:', a-b)
print('a*b:(not dot)', a*b)
print('a/b:', a/b)

# 內積
a = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
]) #(2,3)

b = torch.tensor([
    [9, 8],
    [7, 6],
    [5, 4]
]) #(3,2)
print('a@b:', a@b)


'''
5. 如果只要顯示數值，可轉換成Numpy的張量變數
'''
import numpy as np
array = np.array([
    [1, 2]
]) #(1,2)陣列
tensor = torch.from_numpy(array) # ndarray轉成tensor
print(tensor)


'''
6. reduce_sum: 
用於計算如陣列或張量中所有元素的總和，且可指定維度
通過對所有數值進行加總，來實現將一組數值減少（reduce）到一個單一的數值。
'''
a = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
]) 
print(a.sum(axis=1)) # 對2維(列)進行加總 =[6,15]


'''
7. 稀疏矩陣(sparse matrix)運算: 
指矩陣內只有很少的非零元素。
如果按照正常非是去運算很浪費記憶體(反正都是0)，因此有專門演算法。 

在PyTorch中
 (1) 必須手動將變數移到CPU或GPU。
 (2) 不能一個變數在CPU、另一個在GPU，會ERROR
 (3) 搬到CPU: .cpu() 或 .to('cpu')
 (4) 搬到GPU: .cuda() 或 .to('cuda')
'''
tensor_cpu = torch.tensor([
    [1, 2]
]) # tensor預設在cpu做運算
tensor_gpu = tensor.cuda() # 把tensor搬到cuda(GPU)
print(tensor_gpu)

# 若有多個gpu，則可指定搬到哪
tensor_gpu2 = tensor.to('cuda:0') #但我只有一個gpu
print(tensor_gpu2)

tensor_cpu2 = tensor_gpu2.cpu()
print(tensor_cpu2) # 搬回cpu

'''
8. CPU與GPU不能混和運算
'''
# print( tensor_gpu + tensor_cpu ) # Error 


'''
9. 在GPU與CPU均能執行
'''
if torch.cuda.is_available():
    device = 'cuda'  # 有GPU就設定device是gpu
else:
    device = 'cpu' # 沒有gpu的話就設定cpu
    
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
tensor_gpu.to(device)+tensor_cpu.to(device) # 再把變數放到剛剛決定的device裡


'''
10. PyTorch稀疏矩陣只須設定有值的位置(indices)和數值(values)
如下: i為位置陣列，v為數值陣列
'''
# 定義稀疏矩陣有值的位置(row, column)
i = torch.LongTensor([
    [0, 1, 1], # ([[行],[列]])
    [2, 0, 2]  # (0,2)、(1,0)、(1,2)
])

# 稀疏矩陣的值
v = torch.FloatTensor([3, 4, 5]) #而那三個位置中的值分別是3,4,5

# 定義稀疏矩陣的尺寸(2, 3), 並轉為正常矩陣
a = torch.sparse.FloatTensor(i, v, torch.Size([2, 3])).to_dense()
#(1) 先建立一個稀疏矩陣，並給定: 位置indices, 值value, 矩陣大小torch.Size([])
#(2) 再轉成密集矩陣(正常的矩陣) .to_dense()

print(a)
#  結果是創造一個(2,3)的密集矩陣：
#  (0, 2)元素是3（因為i是[0, 2]且對應的v是3）。
#  (1, 0)元素是4（因為i是[1, 0]且對應的v是4）。
#  (1, 2)元素是5（因為i是[1, 2]且對應的v是5）。
#  其他位置沒有指定值，因此在密集矩陣中被自動填充為0。


'''
11. 稀疏矩陣運算
'''
a = torch.sparse.FloatTensor(i, v, torch.Size([2, 3])) + \
    torch.sparse.FloatTensor(i, v, torch.Size([2, 3])) 
    # \: 允許將一行代碼分成多行來寫 
a.to_dense()
print(a)


'''
12. 直接禁用GPU，但必須一開始就執行，否則無效
'''
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1: 禁用所有CUDA設備
# 檢查GPU以及cuda toolkit 是否存在
print(torch.cuda.is_available()) # 會變成False


'''
13. 若有多張GPU可以指定搬移至某一張GPU
'''
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0'