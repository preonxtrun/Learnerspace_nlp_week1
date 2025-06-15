import numpy as np
arr= np.random.rand(10)
arr = arr*10
print(arr)
arr=np.round(arr, 2)
print(arr)
print(np.min(arr))
print(np.max(arr))
print(np.median(arr))
for i in range(len(arr)):
    if arr[i] < 5:
        arr[i] =arr[i]**2
def numpy_alternate_sort(array):
    sorted_array = np.array([],dtype=array.dtype)
    if len(array)% 2 == 0:
        while len(array)>0:
            sorted_array=np.append(sorted_array,np.min(array))
            sorted_array=np.append(sorted_array,np.max(array))
            array = np.delete(array, np.where(array == np.min(array))[0][0])
            array = np.delete(array, np.where(array == np.max(array))[0][0])
    else:
        while len(array)>1:
            sorted_array=np.append(sorted_array,np.min(array))
            sorted_array=np.append(sorted_array,np.max(array))
            array = np.delete(array, np.where(array == np.min(array))[0][0])
            array = np.delete(array, np.where(array == np.max(array))[0][0])
        sorted_array=np.append(sorted_array,array[0])
    return sorted_array