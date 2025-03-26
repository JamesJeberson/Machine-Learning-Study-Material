Python dictionaries are one of the most **powerful** and **commonly used** data structures. Here's a **detailed rundown** of all the functions, methods, and techniques related to dictionaries. 🚀  

---

## **1. Creating a Dictionary**
### **Using `{}` or `dict()`**
```python
# Method 1: Using curly braces {}
my_dict = {"name": "Alice", "age": 25, "city": "New York"}

# Method 2: Using dict()
my_dict2 = dict(name="Alice", age=25, city="New York")

print(my_dict)  # {'name': 'Alice', 'age': 25, 'city': 'New York'}
```

---

## **2. Accessing Dictionary Values**
### **Using `[]` or `.get()`**
```python
# Using square brackets (Key must exist, or it raises KeyError)
print(my_dict["name"])  # Output: Alice

# Using get() (Returns None or default value if key is missing)
print(my_dict.get("name"))  # Output: Alice
print(my_dict.get("country", "USA"))  # Output: USA (default value)
```

---

## **3. Adding and Updating Key-Value Pairs**
```python
# Adding a new key-value pair
my_dict["country"] = "USA"

# Updating an existing key
my_dict["age"] = 30

print(my_dict)  # {'name': 'Alice', 'age': 30, 'city': 'New York', 'country': 'USA'}
```

---

## **4. Removing Key-Value Pairs**
```python
# Using pop() - removes and returns the value
removed_value = my_dict.pop("age")
print(removed_value)  # Output: 30

# Using del - removes a key-value pair
del my_dict["city"]

# Using popitem() - removes and returns the last inserted key-value pair (Python 3.7+)
last_item = my_dict.popitem()
print(last_item)  # ('country', 'USA')
```

---

## **5. Dictionary Methods**
### **`keys()`, `values()`, `items()`**
```python
print(my_dict.keys())    # Output: dict_keys(['name'])
print(my_dict.values())  # Output: dict_values(['Alice'])
print(my_dict.items())   # Output: dict_items([('name', 'Alice')])
```

### **Looping through Dictionary**
```python
# Iterating over keys
for key in my_dict.keys():
    print(key)

# Iterating over values
for value in my_dict.values():
    print(value)

# Iterating over key-value pairs
for key, value in my_dict.items():
    print(f"{key}: {value}")
```

---

## **6. Checking If a Key Exists**
```python
if "name" in my_dict:
    print("Key exists")

if "salary" not in my_dict:
    print("Key does not exist")
```

---

## **7. Copying a Dictionary**
```python
# Shallow copy
new_dict = my_dict.copy()
new_dict["name"] = "Bob"  # Doesn't change original dictionary

# Using dict() constructor
new_dict2 = dict(my_dict)
```

---

## **8. Merging Two Dictionaries**
```python
# Using update()
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}

dict1.update(dict2)  # Merges dict2 into dict1
print(dict1)  # Output: {'a': 1, 'b': 3, 'c': 4}

# Using dictionary unpacking (Python 3.5+)
merged_dict = {**dict1, **dict2}
print(merged_dict)  # Output: {'a': 1, 'b': 3, 'c': 4}
```

---

## **9. Dictionary Comprehension**
```python
squares = {x: x**2 for x in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# Filtering even values
even_squares = {k: v for k, v in squares.items() if v % 2 == 0}
print(even_squares)  # {2: 4, 4: 16}
```

---

## **10. Default Values in a Dictionary**
### **Using `setdefault()`**
```python
my_dict.setdefault("gender", "Female")  # Adds key if missing
print(my_dict)  # {'name': 'Alice', 'gender': 'Female'}
```

### **Using `defaultdict` from `collections`**
```python
from collections import defaultdict

dd = defaultdict(int)  # Default value is 0 for missing keys
dd["x"] += 5
print(dd)  # {'x': 5}

dd2 = defaultdict(list)  # Default value is an empty list
dd2["y"].append(10)
print(dd2)  # {'y': [10]}
```

---

## **11. Sorting a Dictionary**
### **Sort by Keys**
```python
sorted_by_key = dict(sorted(my_dict.items()))
print(sorted_by_key)
```

### **Sort by Values**
```python
sorted_by_value = dict(sorted(my_dict.items(), key=lambda item: item[1]))
print(sorted_by_value)
```

---

## **12. Nested Dictionary**
```python
nested_dict = {
    "person1": {"name": "Alice", "age": 25},
    "person2": {"name": "Bob", "age": 30}
}

print(nested_dict["person1"]["name"])  # Output: Alice
```

---

## **13. Counting Elements Using Dictionary**
```python
word = "mississippi"
freq = {}

for char in word:
    freq[char] = freq.get(char, 0) + 1

print(freq)  # Output: {'m': 1, 'i': 4, 's': 4, 'p': 2}
```

---

## **14. Dictionary Mapping with `map()`**
```python
names = ["Alice", "Bob", "Charlie"]
name_length = dict(map(lambda x: (x, len(x)), names))
print(name_length)  # {'Alice': 5, 'Bob': 3, 'Charlie': 7}
```

---

## **15. Inverting a Dictionary**
```python
original_dict = {"a": 1, "b": 2, "c": 3}
inverted_dict = {v: k for k, v in original_dict.items()}
print(inverted_dict)  # {1: 'a', 2: 'b', 3: 'c'}
```

---

### **Bonus: Using `collections.OrderedDict` (Preserves Order)**
```python
from collections import OrderedDict

ordered_dict = OrderedDict([("name", "Alice"), ("age", 25), ("city", "New York")])
print(ordered_dict)
```

---

### **Summary Table of Dictionary Methods**
| Method            | Description |
|------------------|-------------|
| `dict.keys()`    | Returns all keys |
| `dict.values()`  | Returns all values |
| `dict.items()`   | Returns all key-value pairs |
| `dict.get(k)`    | Returns value for key `k` (None if missing) |
| `dict.setdefault(k, v)` | Sets default value `v` for key `k` if not present |
| `dict.update(d2)` | Merges `d2` into the dictionary |
| `dict.pop(k)`    | Removes and returns key `k`'s value |
| `dict.popitem()` | Removes last inserted key-value pair |
| `dict.copy()`    | Returns a shallow copy of the dictionary |
| `dict.clear()`   | Removes all elements from dictionary |

---
Here are some commonly asked **Python dictionary** interview problems with solutions. These cover a range of difficulty levels and test key dictionary concepts. 🚀  

---

## **1. Count Character Frequency in a String**
🔹 **Problem:** Given a string, count the frequency of each character.  
🔹 **Solution:** Use a dictionary to store character frequencies.

```python
def char_frequency(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1
    return freq

print(char_frequency("mississippi"))
# Output: {'m': 1, 'i': 4, 's': 4, 'p': 2}
```

---

## **2. Find the Most Frequent Element in a List**
🔹 **Problem:** Given a list, find the element that appears most frequently.  
🔹 **Solution:** Use a dictionary to count occurrences.

```python
def most_frequent(lst):
    freq = {}
    for num in lst:
        freq[num] = freq.get(num, 0) + 1
    return max(freq, key=freq.get)

print(most_frequent([1, 3, 2, 1, 4, 1, 3, 3, 3]))
# Output: 3
```

---

## **3. Merge Two Dictionaries**
🔹 **Problem:** Merge two dictionaries, summing values for common keys.  
🔹 **Solution:** Use `dict.get()` to handle missing keys.

```python
def merge_dicts(d1, d2):
    merged = d1.copy()
    for key, value in d2.items():
        merged[key] = merged.get(key, 0) + value
    return merged

d1 = {"a": 1, "b": 2, "c": 3}
d2 = {"b": 3, "c": 4, "d": 5}

print(merge_dicts(d1, d2))
# Output: {'a': 1, 'b': 5, 'c': 7, 'd': 5}
```

---

## **4. Find the Intersection of Two Dictionaries**
🔹 **Problem:** Given two dictionaries, find the common keys and their values.  
🔹 **Solution:** Use dictionary comprehension.

```python
def dict_intersection(d1, d2):
    return {k: d1[k] for k in d1 if k in d2}

d1 = {"a": 1, "b": 2, "c": 3}
d2 = {"b": 3, "c": 4, "d": 5}

print(dict_intersection(d1, d2))
# Output: {'b': 2, 'c': 3}
```

---

## **5. Reverse a Dictionary (Swap Keys and Values)**
🔹 **Problem:** Swap keys and values in a dictionary.  
🔹 **Solution:** Use dictionary comprehension.

```python
def reverse_dict(d):
    return {v: k for k, v in d.items()}

d = {"a": 1, "b": 2, "c": 3}
print(reverse_dict(d))
# Output: {1: 'a', 2: 'b', 3: 'c'}
```

---

## **6. Group Anagrams**
🔹 **Problem:** Given a list of words, group them by anagram similarity.  
🔹 **Solution:** Use a dictionary with sorted word tuples as keys.

```python
from collections import defaultdict

def group_anagrams(words):
    anagrams = defaultdict(list)
    for word in words:
        sorted_word = tuple(sorted(word))
        anagrams[sorted_word].append(word)
    return list(anagrams.values())

words = ["listen", "silent", "enlist", "rat", "tar", "art"]
print(group_anagrams(words))
# Output: [['listen', 'silent', 'enlist'], ['rat', 'tar', 'art']]
```

---

## **7. Find the First Non-Repeating Character in a String**
🔹 **Problem:** Given a string, find the first character that does not repeat.  
🔹 **Solution:** Use a dictionary to store character counts.

```python
def first_non_repeating(s):
    freq = {}
    for char in s:
        freq[char] = freq.get(char, 0) + 1

    for char in s:
        if freq[char] == 1:
            return char
    return None

print(first_non_repeating("swiss"))
# Output: 'w'
```

---

## **8. Find Two Numbers That Add Up to Target (`Two Sum` Problem)**
🔹 **Problem:** Given a list and a target sum, find two numbers that add up to it.  
🔹 **Solution:** Use a dictionary to store complements.

```python
def two_sum(nums, target):
    num_dict = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_dict:
            return [num_dict[complement], i]
        num_dict[num] = i
    return None

print(two_sum([2, 7, 11, 15], 9))
# Output: [0, 1]
```

---

## **9. Convert Two Lists into a Dictionary**
🔹 **Problem:** Given two lists, convert them into a dictionary.  
🔹 **Solution:** Use `zip()` to pair elements.

```python
def lists_to_dict(keys, values):
    return dict(zip(keys, values))

keys = ["name", "age", "city"]
values = ["Alice", 25, "New York"]

print(lists_to_dict(keys, values))
# Output: {'name': 'Alice', 'age': 25, 'city': 'New York'}
```

---

## **10. Find the Top `K` Frequent Elements**
🔹 **Problem:** Given a list, find the `K` most frequent elements.  
🔹 **Solution:** Use a dictionary and `sorted()`.

```python
from collections import Counter

def top_k_frequent(nums, k):
    count = Counter(nums)
    return [key for key, _ in count.most_common(k)]

print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))
# Output: [1, 2]
```

---

## **11. Remove Duplicate Values from a Dictionary**
🔹 **Problem:** Given a dictionary, remove duplicate values.  
🔹 **Solution:** Use a dictionary with a set.

```python
def remove_duplicates(d):
    seen = set()
    new_dict = {}
    for key, value in d.items():
        if value not in seen:
            new_dict[key] = value
            seen.add(value)
    return new_dict

d = {"a": 1, "b": 2, "c": 2, "d": 3}
print(remove_duplicates(d))
# Output: {'a': 1, 'b': 2, 'd': 3}
```

---

## **12. Flatten a Nested Dictionary**
🔹 **Problem:** Convert a nested dictionary into a flat dictionary with key paths.  
🔹 **Solution:** Use recursion.

```python
def flatten_dict(d, parent_key="", sep="."):
    flat_dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, new_key, sep))
        else:
            flat_dict[new_key] = v
    return flat_dict

nested_dict = {"a": {"b": {"c": 1}}, "d": 2}
print(flatten_dict(nested_dict))
# Output: {'a.b.c': 1, 'd': 2}
```

---
