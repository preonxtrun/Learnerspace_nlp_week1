import numpy as np
arr=np.random.randint(1,50,size=(5,4))
print(arr)
for i in range(arr.shape[0]):
    if arr.shape[1]-1-i>= 0:
        print(arr[i,arr.shape[1]-1-i])
for i in range(arr.shape[0]):
    print(np.max(arr[i,:]))
ar=arr.flatten()
arr1 = np.array([], dtype=type(ar))
for i in ar:
    if i<=np.mean(ar):
        arr1 = np.append(arr1,i)
print(arr1)
def numpy_boundary_traversal(matrix):
    m,n = matrix.shape
    result = np.array([], dtype=matrix.dtype)
    result = np.append(result, matrix[0, :])
    result = np.append(result, matrix[1:m-1,n-1])
    result = np.append(result, matrix[m-1, n-1::-1])
    result = np.append(result, matrix[m-1:0:-1, 0])
    return result
print(numpy_boundary_traversal(arr))