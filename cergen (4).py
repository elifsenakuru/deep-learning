#In this file I have gergen class, fundemental functions, test functions and example function to compare with numpy.
#Numpy is imported for comparison.

import random
import numpy as np
import math
from typing import Union, List
import time 

def cekirdek(sayi: int):
    random.seed(sayi)

def rastgele_dogal(boyut, aralik=(0, 100), dagilim='uniform'):
    if dagilim != 'uniform':
        raise ValueError("Invalid distribution type. Expected 'uniform'.")

    if boyut == ():
        return gergen(random.uniform(aralik[0], aralik[1]))

    veri = []
    for _ in range(boyut[0]):
        if len(boyut) == 1:
            veri.append(random.randint(aralik[0], aralik[1]))
        else:
            veri.append(rastgele_dogal(boyut[1:], aralik=aralik, dagilim=dagilim).veri)

    return gergen(veri)

def rastgele_gercek(boyut, aralik=(0.0, 1.0), dagilim='uniform'):
    if dagilim != 'uniform':
        raise ValueError("Invalid distribution type. Expected 'uniform'.")

    if boyut == ():
        return gergen(random.uniform(aralik[0], aralik[1]))

    veri = []
    for _ in range(boyut[0]):
        if len(boyut) == 1:
            veri.append(random.uniform(aralik[0], aralik[1]))
        else:
            veri.append(rastgele_gercek(boyut[1:], aralik=aralik, dagilim=dagilim).veri)

    return gergen(veri)


class gergen:

    __veri = None #A nested list of numbers representing the data
    D = None # Transpose of data
    __boyut = None #Dimensions of the derivative (Shape)

    def __init__(self, veri=[]):
        #if veri is None, initialize an empty tensor
        if veri is None:
            self.__veri = []
            self.__boyut = None

        #if veri is a number (int or float), initialize a 0D tensor
        if isinstance(veri, (int, float)):
            self.__veri = veri
            self.__boyut = ()

        #if veri is a list, initialize a tensor with the given data
        else:
            self.__veri = veri
            self.__boyut = self.__boyut_hesapla(veri)

    
    @property
    def veri(self):
        return self.__veri
    
    @property
    def boyut(self):
        return self.__boyut

    @property
    def D(self):
        if self._D is None: 
            self._D = self.devrik()
        return self._D
        

    def __boyut_hesapla(self, veri):
        if isinstance(veri, list):
            dimensions = []

            def calculate_boyut(arr):
                if isinstance(arr, list):
                    dimensions.append(len(arr))
                    if arr and isinstance(arr[0], list):
                        calculate_boyut(arr[0])

            calculate_boyut(veri)
            return tuple(dimensions)

        elif isinstance(veri, (int, float)):
            return ()

        elif veri is None:
            return None

        else:
            raise ValueError("Invalid data format for gergen object.")


    def __getitem__(self, index):
        # Indexing for gergen objects
        if isinstance(index, int):
            # singledimensional indexing
            if index < 0 or index >= len(self.__veri):
                raise IndexError("Index out of range")
            return gergen(self.__veri[index])
        elif isinstance(index, tuple):
            # multidimensional indexing
            result = self.__veri
            for idx in index:
                if isinstance(idx, int):
                    if idx < 0 or idx >= len(result):
                        raise IndexError("Index out of range")
                    result = result[idx]
                else:
                    raise TypeError("Invalid index type")
            return gergen(result)
        else:
            raise TypeError("Index must be an integer or a tuple of integers")
        

    def __str__(self):
        def format_list(lst, level=0):
            if not isinstance(lst, list):
                return f"{lst}"
            if len(lst) == 0:
                return "[]"
            if not any(isinstance(el, list) for el in lst):
                return '[' + ' '.join(f"{el}" for el in lst) + ']'
            return ('[' +
                    '\n'.join(format_list(el, level + 1) for el in lst) +
                    ']')

        return format_list(self.__veri)
    

    def __rmul__(self, other):
    # Reuse the __mul__ implementation for scalar multiplication
        if isinstance(other, (int, float)):
            return self.__mul__(other)
        else:
            raise TypeError("Unsupported operand type for multiplication. Expected int or float.")

    def __radd__(self, other):
        # Reuse the __add__ implementation for scalar addition
        if isinstance(other, (int, float)):
            return self.__add__(other)
        else:
            raise TypeError("Unsupported operand type for addition. Expected int or float.")


    def __element_wise_operation(self, other, operation):
        if isinstance(other, gergen):
            if self.__boyut != other.__boyut:
                raise ValueError("Dimension mismatch.")
            return gergen(self.__recursive_operation(self.__veri, other.__veri, operation))
        elif isinstance(other, (int, float)):
            return gergen(self.__recursive_operation(self.__veri, other, operation))
        else:
            raise TypeError("Unsupported operand type.")

    def __recursive_operation(self, data1, data2, operation):
        if isinstance(data1, list) and isinstance(data2, list):
            return [self.__recursive_operation(sub1, sub2, operation) for sub1, sub2 in zip(data1, data2)]
        elif isinstance(data1, list):
            return [self.__recursive_operation(sub1, data2, operation) for sub1 in data1]
        else:
            return operation(data1, data2)

    def __mul__(self, other):
        return self.__element_wise_operation(other, lambda a, b: a * b)

    def __truediv__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")
        return self.__element_wise_operation(other, lambda a, b: a / b)

    def __add__(self, other):
        return self.__element_wise_operation(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self.__element_wise_operation(other, lambda a, b: a - b)


    def uzunluk(self):
    # Returns the total number of elements in the gergen
        total_elements = 1
        for dimension_size in self.__boyut:
            total_elements *= dimension_size
        return total_elements


    def devrik(self):
        #if none we cannot get the transpose
        if self.__veri is None:
            raise ValueError("Cannot get the transpose of an empty gergen.")
        
        #if scalar or 1dimensional with 1 element gergen
        if isinstance(self.__veri, (int, float)) or self.boyut == (1,):
            return self
        
        #if 1D gergen
        if self.boyut == (self.uzunluk(),):
            return gergen([[item] for item in self.__veri])
        
        #if 2D gergen with (n,1)
        elif self.boyut[1] == 1 and len(self.boyut) == 2:
            return gergen([item[0] for item in self.__veri])

        #find indices of the tensor elements
        def indices(boyutlar):
            if not boyutlar:
                yield []
                return
            
            indices_queue = [[]]
            for b in boyutlar:
                new_indices = []
                for j in indices_queue:
                    for i in range(b):
                        new_indices.append(j + [i])
                indices_queue = new_indices
            
            yield from indices_queue
            #print(indices_queue)

        # get the indices of the elements in the original tensor
        indices_before = list(indices(self.boyut))

        #get indices for the transposed tensor
        indices_after = [idx[::-1] for idx in indices_before]
        
        #initialize the new tensor with new boyut
        data = self.boyutlandir(self.boyut[::-1]).listeye()
        
        
        def get_nested(data, idx):
            for i in idx[:-1]:
                data = data[i]
            return data[idx[-1]]

        def set_nested(data, idx, value):
            for i in idx[:-1]:
                data = data[i]
            data[idx[-1]] = value

        for old_index, new_index in zip(indices_before, indices_after):
            value = get_nested(self.__veri, tuple(old_index))
            set_nested(data, new_index, value)
        
        return gergen(data)
    
    
    def sin(self):
        if not self.__veri:
            return gergen()
        return self.__apply_unary_op(math.sin)
        
    def cos(self):
        if not self.__veri:
            return gergen()
        return self.__apply_unary_op(math.cos)

    def tan(self):
        if not self.__veri:
            return gergen()
        return self.__apply_unary_op(math.tan)

    def __apply_unary_op(self, op):
        def apply_op_recursive(data):
            if isinstance(data, (int, float)):
                return op(data)
            elif isinstance(data, list):
                return [apply_op_recursive(subdata) for subdata in data]
            else:
                raise ValueError("Invalid gergen data format.")

        return gergen(apply_op_recursive(self.__veri))
    

    def us(self, n: int):
    #power n of each element of the gergen object .
        if not self.__veri:
            return gergen()
        return self.__apply_unary_op(lambda x: x ** n)

    def log(self):
    #applies the logarithm function to each element of the gergen object.
        if not self.__veri:
            return gergen()
        return self.__apply_unary_op(math.log10)

    def ln(self):
    #applies the natural logarithm function to each element of the gergen object.
        if not self.__veri:
            return gergen()
        return self.__apply_unary_op(math.log)

    #norms should return float 
    def L1(self):
        def l1_norm_recursive(data):
            if isinstance(data, list):
                return sum(l1_norm_recursive(subdata) for subdata in data)
            else:
                return abs(data)

        return l1_norm_recursive(self.__veri)

    def L2(self):
    # Calculates and returns the L2 norm
        def l2_norm_recursive(data):
            if isinstance(data, list):
                return sum(l2_norm_recursive(subdata) for subdata in data)
            else:
                return data ** 2

        return math.sqrt(l2_norm_recursive(self.__veri))

    def Lp(self, p):
    # Calculates and returns the Lp norm, where p should be positive integer
        if not isinstance(p, int) or p < 1:
            raise ValueError("Invalid value for p. Expected a positive integer.")
        def lp_norm_recursive(data):
            if isinstance(data, list):
                return sum(lp_norm_recursive(subdata) for subdata in data)
            else:
                return abs(data) ** p

        return lp_norm_recursive(self.__veri) ** (1 / p)
    

    def listeye(self):
    #converts the gergen object into a list or a nested list, depending on its dimensions.
        return self.__veri

    def duzlestir(self):
        def flatten_recursive(data):
            if isinstance(data, list):
                return [element for sublist in data for element in flatten_recursive(sublist)]
            else:
                return [data]

        flattened_data = flatten_recursive(self.__veri)
        return gergen(flattened_data)

    def boyutlandir(self, yeni_boyut: tuple) -> 'gergen':
        # Check if yeni_boyut is a tuple
        if not isinstance(yeni_boyut, tuple):
            raise TypeError("yeni_boyut must be a tuple")

        # Check if the product of the new dimensions matches the total number of elements
        total_elements = self.uzunluk()
        print(total_elements)
        print(math.prod(yeni_boyut))
        if total_elements != math.prod(yeni_boyut):
            raise ValueError("The product of the new dimensions must equal the total number of elements")

        # flatten the tensor to a 1D list
        flattened = self.duzlestir().veri

        # helper function to reshape the flattened list
        def reshape(lst, shape):
            if len(shape) == 1:
                return lst
            size = math.prod(shape[1:])
            return [reshape(lst[i * size: (i + 1) * size], shape[1:]) for i in range(shape[0])]

        # reshape the flattened list to the new shape
        reshaped_data = reshape(flattened, yeni_boyut)
        return gergen(reshaped_data)


    def ic_carpim(self, other):
    #Calculates the inner (dot) product of this gergen object with another.
        #if either are them gergens raise error
        if not isinstance(other, gergen):
            raise TypeError("Unsupported operand type for inner product. Expected gergen.")
        #for 1D gergen
        if len(self.__boyut) == 1 and len(other.__boyut) == 1:
            if len(self.__veri) != len(other.__veri):
                raise ValueError("Invalid dimensions for inner product. Expected same length.")
            return sum(a * b for a, b in zip(self.__veri, other.__veri))
        #for 2D gergen
        if len(self.__boyut) == 2 and len(other.__boyut) == 2:
            if self.__boyut[1] != other.__boyut[0]:
                raise ValueError("Invalid dimensions for inner product. Expected (m x n) * (n x p).")
            result = []
            for i in range(self.__boyut[0]):
                row = []
                for j in range(other.__boyut[1]):
                    element = 0
                    for k in range(self.__boyut[1]):
                        element += self.__veri[i][k] * other.__veri[k][j]
                    row.append(element)
                result.append(row)
            return gergen(result)

    def dis_carpim(self, other):
        if not isinstance(other, gergen):
            raise TypeError("Unsupported operand type for outer product. Expected gergen.")

        if len(self.__boyut) > 2 or len(other.__boyut) > 2:
            raise ValueError("Both operands must be 1-D or 2-D arrays to compute the outer product.")

        result = []
        if len(self.__boyut) == 1 and len(other.__boyut) == 1:
            # Both are 1-D arrays
            for a in self.__veri:
                row = [a * b for b in other.__veri]
                result.append(row)
        else:
            # At least one operand is a 2-D array
            for row_a in self.__veri:
                row_result = []
                for row_b in other.__veri:
                    element = sum(a * b for a, b in zip(row_a, row_b))
                    row_result.append(element)
                result.append(row_result)

        return gergen(result)

        

    def _sum_along_axis(self, data, axis):
        if axis == 0:
            if isinstance(data[0], list):
                return [self._sum_along_axis([x[i] for x in data], 0) for i in range(len(data[0]))]
            else:
                return sum(data)
        else:
            return [self._sum_along_axis(sublist, axis - 1) for sublist in data]
    

    def topla(self, eksen=None):
        if not isinstance(eksen, (int, type(None))):
            raise TypeError("eksen must be an integer or None")

        if eksen is None:
            return sum(self.duzlestir().veri)

        elif eksen >= 0:
            try:
                return gergen(self._sum_along_axis(self.__veri, eksen))
            except IndexError:
                raise ValueError(" eksen is out of bounds")

        else:
            raise ValueError("Specified eksen is out of bounds")
        
    def ortalama(self, eksen=None):
        if not isinstance(eksen, (int, type(None))):
            raise TypeError("eksen must be an integer or None")

        def count_elements(data, axis):
            if axis == 0:
                return len(data)
            else:
                return [count_elements(sublist, axis - 1) for sublist in data]

        if eksen is None:
            total_sum = self.topla()
            total_elements = self.uzunluk()
            return total_sum / total_elements
        elif eksen >= 0:
            try:
                sums = self.topla(eksen).veri
                counts = count_elements(self.veri, eksen)
                if isinstance(sums, list):
                    if isinstance(counts, int):
                        mean = [sum(x) / counts for x in zip(*[sums])]
                    else:
                        mean = [x / c for x, c in zip(sums, counts)]
                else:
                    mean = sums / counts
                return gergen(mean)
            except IndexError:
                raise ValueError("Specified eksen is out of bounds")
        else:
            raise ValueError("Specified eksen is out of bounds")


def test1():

    veri = [[1, 2, 3], [4, 5, 6]]
    tensor = gergen(veri)
    print("Tensor data:", tensor._gergen__veri)
    print("Tensor size:", tensor._gergen__boyut)

    # Test the init method with no data
    #empty_tensor = gergen()
    #print("Empty tensor data:", empty_tensor._gergen__veri)
    #print("Empty tensor size:", empty_tensor._gergen__boyut)

    # Test the boyut_hesapla method
    veri = [[1, 2, 3], [4, 5, 6]]
    gergen_instance = gergen()
    size = gergen_instance._gergen__boyut_hesapla(veri)
    print("Size of tensor:", size)

    # Test the __getitem__ method
    veri = [[1, 2, 3], [4, 5, 6]]
    tensor = gergen(veri)
    print("Element at index (1, 1):", tensor[1][1])

    # Test the __str__ method
    veri = [[1, 2, 3], [4, 5, 6]]
    tensor = gergen(veri)
    print("Tensor as string:", tensor)

    # Test the __mul__ method
    veri = [[1, 2, 3], [4, 5, 6]]
    tensor = gergen(veri)
    result = tensor * 2
    print("Result of multiplication:", result)

    #test multiplication of two tensors
    # veri1 = [[1, 2, 3], [4, 5, 6]]
    # veri2 = [[1, 2], [3, 4], [5, 6]]
    # tensor1 = gergen(veri1)
    # tensor2 = gergen(veri2)
    # result = tensor1*tensor2
    # print("Result of multiplication:", result)

    # Test the __truediv__ method
    veri = [[1, 2, 3], [4, 5, 6]]
    tensor = gergen(veri)
    result = tensor.__truediv__(2)
    print("Result of division:", result)

    #test the truediv method with another tensor
    veri1 = [[1, 2, 3], [4, 5, 6]]
    veri2 = [[2, 3, 4], [5, 6, 7]]
    tensor1 = gergen(veri1)
    tensor2 = gergen(veri2)
    result = tensor1.devrik()
    print("Result of division:", result)

    # Create a 4D tensor with derivative elements each element should be pi
    tensor_data = [[[[math.pi, math.pi], [math.pi, math.pi]], [[math.pi, math.pi], [math.pi, math.pi]]],
                [[[math.pi, math.pi], [math.pi, math.pi]], [[math.pi, math.pi], [math.pi, math.pi]]]]

    tensor = gergen(tensor_data)

    # Apply sine function to the tensor
    sine_tensor = tensor.sin()

    print(sine_tensor)
    print(math.sin(math.pi))


def test2():

    #TESTS
    #tests for boyut
    #compare with numpy
    tensor1 = gergen([[1, 2, 3], [4, 5, 6]])
    arr = np.array(tensor1.veri)
    print(arr.shape)
    print(tensor1.boyut)

    tensor2 = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    arr = np.array(tensor2.veri)
    print(arr.shape)
    print(tensor2.boyut)

    tensor3 = gergen([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])
    arr = np.array(tensor3.veri)
    print(arr.shape)
    print(tensor3.boyut)

    #TESTS
    #tests for getitem
    #compare with numpy
    tensor1 = gergen([[1, 2, 3], [4, 5, 6]])
    arr = np.array(tensor1.veri)
    print(arr[0])
    print(tensor1[0])

    tensor2 = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    arr = np.array(tensor2.veri)
    print(arr[1][0])
    print(tensor2[1][0])

    tensor3 = gergen([[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]])
    arr = np.array(tensor3.veri)
    print(arr[0][1][0][1])
    print(tensor3[0][1][0][1])

    #TESTS
    #tests for mul
    #compare with numpy

    tensor1 = gergen([[1, 2, 3], [4, 5, 6]])
    arr = np.array(tensor1.veri)
    print(arr*2)
    print(tensor1*2)
    print(3*arr)


    tensor2 = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    tensor3 = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    arr2 = np.array(tensor2.veri)
    arr3 = np.array(tensor3.veri)
    print(arr2*arr3)
    print(tensor2*tensor3)

    #test for scalar gergens
    tensor1 = gergen(5)
    arr = np.array(5)
    print(arr*2)
    print(tensor1*2)

    #TESTS
    #tests for add
    #compare with numpy

    tensor1 = gergen([[1, 2, 3], [4, 5, 6]])
    arr = np.array(tensor1.veri)
    print(arr+2)
    print(tensor1+2)

    tensor2 = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    tensor3 = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    arr2 = np.array(tensor2.veri)
    arr3 = np.array(tensor3.veri)
    print(arr2+arr3)
    print(tensor2+ tensor3)

    #test for scalar gergens
    tensor1 = gergen(5)
    arr = np.array(5)
    print(arr+2)
    print(tensor1+2)

    #TESTS
    #tests for subtraction
    #compare with numpy

    tensor1 = gergen([[1, 2, 3], [4, 5, 6]])
    arr = np.array(tensor1.veri)
    print(arr-2)
    print(tensor1-2)

    tensor2 = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    tensor3 = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    arr2 = np.array(tensor2.veri)
    arr3 = np.array(tensor3.veri)
    print(arr2-arr3)
    print(tensor2- tensor3)

    #test for scalar gergens
    tensor1 = gergen(5)
    arr = np.array(5)
    print(arr-2)
    print(tensor1-2)

    #TESTS
    #tests for true_dix
    #compare with numpy

    tensor1 = gergen([[1, 2, 3], [4, 5, 6]])
    arr = np.array(tensor1.veri)
    print(arr/2)
    print(tensor1/2)

    tensor2 = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    tensor3 = gergen([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    arr2 = np.array(tensor2.veri)
    arr3 = np.array(tensor3.veri)
    print(arr2/arr3)
    print(tensor2/ tensor3)

    #test for scalar gergens
    tensor1 = gergen(5)
    arr = np.array(5)
    print(arr/2)
    print(tensor1/2)

    tensor1 = rastgele_dogal((2,3,4))

    arr = np.array(tensor1.listeye())
    print(arr)
    t_arr = arr.transpose()
    print(t_arr)
    print(str(tensor1.devrik()))


def test3():
    #write test for ortalama
    tensor = rastgele_dogal((4, 2, 3))
    '''print(tensor.topla())
    print(tensor.topla(0))'''
    print(tensor)
    arr = np.array(tensor.veri)
    print(np.transpose(arr))
    print(tensor.devrik())
    '''print(np.sum(arr))
    print(np.sum(arr, axis=0))'''
    gregen = gergen([1])
    print(gregen.boyut)
    gregen = gergen([1,2,3])
    print(gregen.boyut)
    print(gregen.uzunluk())
    print(gregen.devrik())
    #if 2D gergen with (n,1)

    tensor = rastgele_dogal((3, 2))
    print(tensor)
    print(tensor.devrik())

    print(tensor.boyutlandir((3, 2)).veri)

    print(3*tensor)
    print(tensor*3)
    print(4+tensor)
    print(tensor.ortalama(0))
    arr = np.array(tensor.veri)
    print(np.mean(arr, axis=0))

    tensor =rastgele_dogal((3,3,3,4))
    #print(tensor.__str__())
    print(tensor.boyut)
    print(tensor[0])
    print(tensor[0][0][0][0])
    print(tensor.__getitem__((0,0,0,0)))
    print(tensor.__getitem__((0,0)))
    print(tensor.__getitem__(0))
    print(tensor[0])
    print(tensor)
    a = [1,2,3,4,5,6,7,8,9]
    b = [20,30,40,50,60]
    zip(a,b)
    print(list(zip(a,b)))
    print(tensor.uzunluk())

    boyut = ()
    aralik = (0, 10)
    g0 = rastgele_gercek(boyut, aralik)
    print(g0)

    boyut = ()
    aralik = (0, 10)
    g0 = rastgele_gercek(boyut, aralik)
    print(g0)

    boyut = ()
    aralik = (0, 10)
    g0 = rastgele_gercek(boyut, aralik)
    print(g0)

    g1 = gergen([[1, 2, 3], [4, 5, 6]])
    print(g1)
    print(type(g1))
    print(g1.boyut)

    g2 = (rastgele_dogal((3, 1)))
    print(g2)
    print(type(g2))

    print(g2.boyut)

    tensor = rastgele_dogal((3, 1))
    print(tensor)
    print(tensor.devrik())

    print(tensor.boyutlandir((3, 1)).veri)

    print(3*tensor)
    print(tensor*3)
    print(4+tensor)

def test4():
    def test_devrik():
        print("Testing devrik:")
        a = gergen([[1, 2], [3, 4]])
        assert a.devrik().veri == [[1, 3], [2, 4]], "Failed on 2x2 matrix"
        
        b = gergen([1, 2, 3])
        assert b.devrik().veri == [[1], [2], [3]], "Failed on 1D vector"
        
        c = gergen(5)
        assert c.devrik().veri == 5, "Failed on scalar"
        
        print("Passed all tests for devrik.")

    def test_ic_carpim():
        print("Testing ic_carpim:")
        a = gergen([1, 2, 3])
        b = gergen([4, 5, 6])
        assert a.ic_carpim(b) == 32, "Failed on 1D vectors"
        
        c = gergen([[1, 2], [3, 4]])
        d = gergen([[5, 6], [7, 8]])
        assert c.ic_carpim(d).veri == [[19, 22], [43, 50]], "Failed on 2x2 matrices"
        
        print("Passed all tests for ic_carpim.")

    def test_dis_carpim():
        print("Testing dis_carpim:")
        a = gergen([1, 2])
        b = gergen([3, 4])
        assert a.dis_carpim(b).veri == [[3, 4], [6, 8]], "Failed on 1D vectors"
        
        print("Passed all tests for dis_carpim.")

    def test_topla():
        print("Testing topla:")
        a = gergen([[1, 2], [3, 4]])
        assert a.topla() == 10, "Failed on total sum"
        assert a.topla(eksen=0).veri == [4, 6], "Failed on column-wise sum"
        assert a.topla(eksen=1).veri == [3, 7], "Failed on row-wise sum"
        
        b = gergen([1, 2, 3])
        assert b.topla() == 6, "Failed on 1D vector"
        
        print("Passed all tests for topla.")

    def test_ortalama():
        print("Testing ortalama:")
        a = gergen([[1, 2], [3, 4]])
        assert a.ortalama() == 2.5, "Failed on overall average"
        assert a.ortalama(eksen=0).veri == [2.0, 3.0], "Failed on column-wise average"
        assert a.ortalama(eksen=1).veri == [1.5, 3.5], "Failed on row-wise average"
        
        b = gergen([1, 2, 3])
        assert b.ortalama() == 2.0, "Failed on 1D vector"
        
        print("Passed all tests for ortalama.")

    # Run all tests
    test_devrik()
    test_ic_carpim()
    test_dis_carpim()
    test_topla()
    test_ortalama()


def example_1():
    boyut = (64, 64)
    g1 = rastgele_gercek(boyut)
    g2 = rastgele_gercek(boyut)

    start = time.time()
    result1 = g1.ic_carpim(g2)
    end = time.time()

    start_np = time.time()
    result2 = np.dot(np.array(g1.veri), np.array(g2.veri))
    end_np = time.time()

    if np.allclose(np.array(result1.veri), result2, atol=1e-6):
        print("The results are close enough.")
    else:
        print("The results are not close enough.")

    print("Time taken for gergen:", end - start)
    print("Time taken for NumPy:", end_np - start_np)


def example_2():
    boyut = (4, 16, 16, 16)
    a = rastgele_gercek(boyut)
    b = rastgele_gercek(boyut)
    c = rastgele_gercek(boyut)

    start = time.time()
    result_gergen = (a * b + a * c + b * c).ortalama()
    print(result_gergen)
    end = time.time()

    start_np = time.time()
    a_np = np.array(a.veri)
    b_np = np.array(b.veri)
    c_np = np.array(c.veri)
    result_np = np.mean(a_np * b_np + a_np * c_np + b_np * c_np, axis=None)
    print(result_np)
    end_np = time.time()

    if np.allclose(result_gergen, result_np, atol=1e-6):
        print("The results are close enough.")
    else:
        print("The results are not close enough.")

        print("Time taken for gergen:", end - start)
        print("Time taken for NumPy:", end_np - start_np)


def example_3():
    boyut = (4, 16, 16, 16)
    a = rastgele_gercek(boyut)
    b = rastgele_gercek(boyut)
    c = rastgele_gercek(boyut)

    start = time.time()
    result_gergen = ((a.sin() + b.cos()).ln()).us(2)
    end = time.time()

    start_np = time.time()
    a_np = np.array(a.veri)
    b_np = np.array(b.veri)
    c_np = np.array(c.veri)
    result_np = (np.log(np.sin(a_np) + np.cos(b_np))) ** 2
    end_np = time.time()

    if np.allclose(np.array(result_gergen.veri), result_np, atol=1e-6):
        print("The results are close enough.")
    else:
        print("The results are not close enough.")

    print("Time taken for gergen:", end - start)
    print("Time taken for NumPy:", end_np - start_np)

def run_tests():
    test1()
    test2()
    test3()
    test4()

def run_numpy_comparison():
    example_1()
    example_2()
    example_3()

def __main__():
    run_tests()
    run_numpy_comparison()
