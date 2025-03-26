Python lists are one of the most commonly used data structures, and they come with a variety of functions, methods, and techniques. Here's a comprehensive guide:

---

## **1. Creating Lists**
```python
# Empty list
lst = []

# List with elements
lst = [1, 2, 3, 4, 5]

# Mixed data types
lst = [1, "hello", 3.14, True]

# Nested lists
nested_lst = [[1, 2, 3], [4, 5, 6]]

# Using list constructor
lst = list((1, 2, 3, 4, 5))  # Converts tuple to list

# Using list comprehension
squares = [x**2 for x in range(5)]
```

---

## **2. Accessing Elements**
```python
# Indexing (0-based)
print(lst[0])  # First element
print(lst[-1]) # Last element

# Slicing
print(lst[1:4])  # Elements from index 1 to 3
print(lst[:3])   # First 3 elements
print(lst[::2])  # Every second element
print(lst[::-1]) # Reverse list
```

---

## **3. Modifying Lists**
```python
# Changing element at index
lst[0] = 10

# Adding elements
lst.append(6)         # Adds to the end
lst.insert(2, 100)    # Inserts at index 2

# Extending a list
lst.extend([7, 8, 9]) # Merges lists

# Removing elements
lst.remove(3)   # Removes first occurrence of 3
popped = lst.pop(2) # Removes and returns element at index 2
lst.clear()     # Removes all elements
```

---

## **4. List Methods**
```python
lst = [3, 1, 4, 1, 5, 9, 2]

# Sorting
lst.sort()         # Sorts in-place
lst.sort(reverse=True) # Descending order
sorted_lst = sorted(lst)  # Returns new sorted list

# Reversing
lst.reverse()      # Reverses in-place
reversed_lst = list(reversed(lst))  # Returns new reversed list

# Counting occurrences
print(lst.count(1)) # Number of times 1 appears

# Finding index
print(lst.index(5)) # First occurrence of 5
```

---

## **5. Copying Lists**
```python
lst_copy = lst.copy()       # Creates a shallow copy
lst_copy = lst[:]           # Another way to copy
lst_copy = list(lst)        # Another way to copy
import copy
deep_copy = copy.deepcopy(lst) # Deep copy (for nested lists)
```

---

## **6. List Comprehension**
```python
squares = [x**2 for x in range(10)] 
even_numbers = [x for x in range(10) if x % 2 == 0]
nested_list = [[x for x in range(3)] for y in range(3)]
```

---

## **7. Membership and Iteration**
```python
# Checking if element exists
print(5 in lst)  # True or False

# Looping through a list
for item in lst:
    print(item)

# Enumerate (index + value)
for index, value in enumerate(lst):
    print(index, value)
```

---

## **8. Joining and Splitting**
```python
words = ["Hello", "World"]
sentence = " ".join(words)  # "Hello World"

string = "apple,banana,cherry"
fruit_list = string.split(",") # ['apple', 'banana', 'cherry']
```

---

## **9. Converting Other Data Structures to Lists**
```python
tuple_to_list = list((1, 2, 3))
string_to_list = list("hello") # ['h', 'e', 'l', 'l', 'o']
set_to_list = list({1, 2, 3})
dict_keys_to_list = list({'a': 1, 'b': 2}.keys())
dict_values_to_list = list({'a': 1, 'b': 2}.values())
```

---

## **10. Useful `itertools` Functions (Advanced)**
```python
from itertools import permutations, combinations, product

lst = [1, 2, 3]

# All permutations
print(list(permutations(lst, 2))) 

# All combinations
print(list(combinations(lst, 2)))

# Cartesian product
print(list(product(lst, repeat=2)))
```

---

## **11. Using `map`, `filter`, `reduce`**
```python
from functools import reduce

# Mapping function to all elements
squared = list(map(lambda x: x**2, lst))

# Filtering elements
even_numbers = list(filter(lambda x: x % 2 == 0, lst))

# Reducing a list to a single value
sum_of_list = reduce(lambda x, y: x + y, lst)
```

Here are some **commonly asked Python interview problems** related to **lists**, along with their solutions.  

---

## **1. Find the Second Largest Element in a List**
**Problem:** Given a list of numbers, find the second largest element.  

### **Solution**
```python
def second_largest(lst):
    unique_nums = list(set(lst))  # Remove duplicates
    if len(unique_nums) < 2:
        return None  # No second largest element exists
    unique_nums.sort(reverse=True)  # Sort in descending order
    return unique_nums[1]

# Example
lst = [10, 20, 4, 45, 99, 99, 20]
print(second_largest(lst))  # Output: 45
```
**Optimized Approach:** Instead of sorting, keep track of the largest and second-largest in one pass (**O(n) time**).
```python
def second_largest(lst):
    first = second = float('-inf')
    for num in lst:
        if num > first:
            second, first = first, num
        elif first > num > second:
            second = num
    return second if second != float('-inf') else None

print(second_largest([10, 20, 4, 45, 99]))  # Output: 45
```
---

## **2. Find the Missing Number in an Array**
**Problem:** Given a list containing `n` distinct numbers from `1` to `n+1`, find the missing number.

### **Solution**
```python
def missing_number(lst):
    n = len(lst) + 1  # Total numbers expected
    expected_sum = n * (n + 1) // 2  # Sum of first n natural numbers
    actual_sum = sum(lst)
    return expected_sum - actual_sum

print(missing_number([1, 2, 4, 6, 3, 7, 8]))  # Output: 5
```
**Optimized Approach using XOR:**  
```python
def missing_number_xor(lst):
    n = len(lst) + 1
    xor_all = 0
    xor_list = 0

    for i in range(1, n+1):
        xor_all ^= i  # XOR of all numbers from 1 to n

    for num in lst:
        xor_list ^= num  # XOR of list elements

    return xor_all ^ xor_list  # Missing number

print(missing_number_xor([1, 2, 4, 6, 3, 7, 8]))  # Output: 5
```
---

## **3. Find All Pairs That Sum to a Target**
**Problem:** Given a list of integers, find all unique pairs that sum up to a target.

### **Solution**
```python
def find_pairs(lst, target):
    seen = set()
    pairs = set()
    
    for num in lst:
        diff = target - num
        if diff in seen:
            pairs.add((min(num, diff), max(num, diff)))  # Store in sorted order to avoid duplicates
        seen.add(num)

    return list(pairs)

print(find_pairs([2, 4, 3, 7, 8, 9, -1, 5], 8))
# Output: [(3, 5), (-1, 9)]
```
---

## **4. Find Duplicates in a List**
**Problem:** Given a list, find all duplicate elements.

### **Solution**
```python
from collections import Counter

def find_duplicates(lst):
    counts = Counter(lst)  # Counts occurrences of each element
    return [key for key, val in counts.items() if val > 1]

print(find_duplicates([1, 2, 3, 4, 5, 2, 3, 6, 7, 8, 1]))  
# Output: [1, 2, 3]
```
---

## **5. Move Zeros to End of List**
**Problem:** Given a list, move all zeros to the end while maintaining the relative order of non-zero elements.

### **Solution**
```python
def move_zeros(lst):
    non_zeros = [num for num in lst if num != 0]
    return non_zeros + [0] * (len(lst) - len(non_zeros))

print(move_zeros([0, 1, 0, 3, 12]))
# Output: [1, 3, 12, 0, 0]
```
**Optimized In-Place Approach:**
```python
def move_zeros(lst):
    pos = 0
    for i in range(len(lst)):
        if lst[i] != 0:
            lst[pos], lst[i] = lst[i], lst[pos]
            pos += 1
    return lst

print(move_zeros([0, 1, 0, 3, 12]))  
# Output: [1, 3, 12, 0, 0]
```
---

## **6. Rotate a List to the Right by K Steps**
**Problem:** Given a list, rotate it to the right by `k` steps.

### **Solution**
```python
def rotate_list(lst, k):
    k = k % len(lst)  # Handle cases where k > len(lst)
    return lst[-k:] + lst[:-k]

print(rotate_list([1, 2, 3, 4, 5], 2))
# Output: [4, 5, 1, 2, 3]
```
---

## **7. Find the Intersection of Two Lists**
**Problem:** Given two lists, return their intersection.

### **Solution**
```python
def list_intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))  # Using set intersection

print(list_intersection([1, 2, 3, 4], [2, 3, 5, 7]))
# Output: [2, 3]
```
---

## **8. Find the First Non-Repeating Element**
**Problem:** Find the first element in the list that appears only once.

### **Solution**
```python
from collections import Counter

def first_unique(lst):
    counts = Counter(lst)
    for num in lst:
        if counts[num] == 1:
            return num
    return None

print(first_unique([4, 5, 1, 2, 0, 4, 5, 2]))  
# Output: 1
```
---

## **9. Merge Two Sorted Lists**
**Problem:** Given two sorted lists, merge them into a single sorted list.

### **Solution**
```python
def merge_sorted(lst1, lst2):
    return sorted(lst1 + lst2)  # Simple approach using sorting

print(merge_sorted([1, 3, 5], [2, 4, 6]))
# Output: [1, 2, 3, 4, 5, 6]
```
**Optimized Two-Pointer Approach (O(n)):**
```python
def merge_sorted(lst1, lst2):
    i, j = 0, 0
    merged = []
    
    while i < len(lst1) and j < len(lst2):
        if lst1[i] < lst2[j]:
            merged.append(lst1[i])
            i += 1
        else:
            merged.append(lst2[j])
            j += 1
    
    merged.extend(lst1[i:])
    merged.extend(lst2[j:])
    
    return merged

print(merge_sorted([1, 3, 5], [2, 4, 6]))
# Output: [1, 2, 3, 4, 5, 6]
```
---

## **10. Find the Longest Consecutive Sequence**
**Problem:** Given an unsorted list, find the length of the longest consecutive sequence.

### **Solution**
```python
def longest_consecutive(lst):
    num_set = set(lst)
    max_length = 0

    for num in lst:
        if num - 1 not in num_set:
            current_num = num
            count = 1

            while current_num + 1 in num_set:
                current_num += 1
                count += 1

            max_length = max(max_length, count)

    return max_length

print(longest_consecutive([100, 4, 200, 1, 3, 2]))
# Output: 4 (sequence: 1,2,3,4)
```